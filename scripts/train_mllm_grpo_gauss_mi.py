#!/usr/bin/env python
"""
MLLM Early-Disentangle (GaussMI) GRPO Training Script

Stacked LoRA scheme:
  Scenario A: No SFT LoRA -> create grpo adapter (trainable), ref uses base model
  Scenario B: With SFT LoRA -> load sft adapter (frozen) + create grpo adapter (trainable)
              Policy = base + sft + grpo, Reference = base + sft

Reward: three-component structure
  - rule_reward: format/steps/modality, each 1/3
  - q_similarity: Qwen3-Reranker semantic similarity between generated and GT questions
  - pid_reward: continuous per-sample PID synergy score

Usage:
  CUDA_VISIBLE_DEVICES=0,1 deepspeed scripts/train_mllm_grpo_gauss_mi.py \\
    --config config/mllm_grpo_gauss_mi.yaml \\
    --deepspeed config/deepspeed_grpo.json
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import re
import yaml
import random
import numpy as np
import torch
import torch.nn.functional as F
import deepspeed
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer as HFAutoTokenizer, AutoModelForCausalLM

from src.models.mllm_backbone import build_mllm_backbone, SFT_ADAPTER, GRPO_ADAPTER
from src.models.mllm_early_disentangle_gauss_mi_model import MLLMEarlyDisentangleGaussMIModel
from src.data.mllm_dataset import MLLMDataset, mllm_collate_fn
from src.data.mllm_ablation_no_cot_dataset import MLLMNoCotDataset
from src.reward.pid_reward_model import PIDScorerWrapper, vMFScorerWrapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TRAINABLE_MODULES = ['shared_dual', 'visual_private', 'text_private', 'segment_embedding']


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_distributed():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def deep_memory_cleanup(rank, world_size, sync_barrier=True):
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    if sync_barrier and world_size > 1:
        torch.distributed.barrier()


def compute_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """group-relative advantages, rewards: [B, G]"""
    advantages = rewards - rewards.mean(dim=1, keepdim=True)
    advantages = advantages / (rewards.std(dim=1, keepdim=True, unbiased=False) + 1e-8)
    return advantages


def extract_question_ids(generated_ids, tokenizer, pad_id, return_start_positions=False):
    """Extract [Question] tag and subsequent token IDs from generated sequence"""
    q_tag_ids = tokenizer.encode("[Question]", add_special_tokens=False)
    q_tag_len = len(q_tag_ids)
    q_tag_tensor = torch.tensor(q_tag_ids, dtype=generated_ids.dtype)

    batch_size, gen_len = generated_ids.shape
    question_ids_list = []
    start_positions = []

    for b in range(batch_size):
        ids = generated_ids[b]
        ids_list = ids.tolist()
        start_pos = -1

        for i in range(gen_len - q_tag_len + 1):
            if (ids[i:i + q_tag_len] == q_tag_tensor.to(ids.device)).all():
                start_pos = i
                break

        if start_pos == -1:
            full_text = tokenizer.decode(ids_list, skip_special_tokens=False)
            q_char_pos = full_text.find("[Question]")
            if q_char_pos != -1:
                lo, hi = 0, gen_len
                while lo < hi:
                    mid = (lo + hi) // 2
                    prefix_len = len(tokenizer.decode(ids_list[:mid], skip_special_tokens=False))
                    if prefix_len <= q_char_pos:
                        lo = mid + 1
                    else:
                        hi = mid
                for j in range(max(0, lo - 3), min(lo + 1, gen_len)):
                    suffix = tokenizer.decode(ids_list[j:], skip_special_tokens=False)
                    if "[Question]" in suffix:
                        start_pos = j
                        break

        if start_pos == -1:
            question_ids_list.append(
                torch.tensor([pad_id], dtype=generated_ids.dtype, device=generated_ids.device)
            )
            start_positions.append(gen_len)
        else:
            question_ids_list.append(ids[start_pos:])
            start_positions.append(start_pos)

    max_q_len = max(len(ids) for ids in question_ids_list)
    question_ids = torch.full(
        (batch_size, max_q_len), pad_id,
        dtype=generated_ids.dtype, device=generated_ids.device
    )
    for b, ids_t in enumerate(question_ids_list):
        question_ids[b, :len(ids_t)] = ids_t

    if return_start_positions:
        q_starts = torch.tensor(start_positions, dtype=torch.long, device=generated_ids.device)
        return question_ids, q_starts
    return question_ids


def load_pretrained_checkpoint(model: MLLMEarlyDisentangleGaussMIModel, config: dict, rank: int):
    """Load SFT pretrained checkpoint (disentangle modules)"""
    ckpt_path = config['training'].get('load_pretrained')
    if not ckpt_path or not os.path.exists(ckpt_path):
        return
    if rank == 0:
        logger.info(f"Loading SFT checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    for mname in TRAINABLE_MODULES:
        if mname in checkpoint:
            getattr(model, mname).load_state_dict(checkpoint[mname])
            if rank == 0:
                logger.info(f"  ✓ {mname}")


def configure_freezing(model: MLLMEarlyDisentangleGaussMIModel, config: dict, rank: int):
    """GRPO phase freezing strategy"""
    training_cfg = config['training']

    if training_cfg.get('freeze_backbone_llm', True):
        for name, param in model.backbone.named_parameters():
            if 'lora' not in name.lower():
                param.requires_grad = False
        if rank == 0:
            logger.info("  - Backbone (non-LoRA): frozen")

    freeze_map = {
        'freeze_shared_dual': 'shared_dual',
        'freeze_visual_private': 'visual_private',
        'freeze_text_private': 'text_private',
        'freeze_segment_emb': 'segment_embedding',
    }
    for cfg_key, mname in freeze_map.items():
        if training_cfg.get(cfg_key, True):
            for p in getattr(model, mname).parameters():
                p.requires_grad = False
            if rank == 0:
                logger.info(f"  - {mname}: frozen")

    if rank == 0:
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable params: {num_trainable:,} / {num_total:,} ({num_trainable/num_total*100:.2f}%)")


def setup_stacked_lora(model: MLLMEarlyDisentangleGaussMIModel, config: dict, rank: int):
    """
    Dual LoRA stacking initialization (sft frozen + grpo trainable)

    Three forward states:
      Policy:    active = [sft, grpo] -> W + dW_sft + dW_grpo
      Reference: active = [sft]      -> W + dW_sft
      Base:      disable all          -> W

    Scenario A: No SFT LoRA -> only create grpo adapter, Reference = base
    Scenario B: PEFT directory -> load as sft adapter (frozen) + create grpo adapter (trainable)
    Scenario C: .pt file -> load SFT weights into sft adapter (frozen) + create grpo adapter (trainable)

    Returns use_base_as_ref: bool
    """
    from peft import PeftModel as PM, LoraConfig, get_peft_model

    sft_lora_path = config['training'].get('load_pretrained_lora')
    model_cfg = config['model']

    # SFT LoRA config (for creating sft adapter structure)
    sft_lora_cfg = model_cfg.get('lora') or {
        'r': 16, 'lora_alpha': 32,
        'lora_dropout': 0.0,
        'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        'bias': "none",
    }
    sft_lora_config = LoraConfig(
        r=sft_lora_cfg['r'],
        lora_alpha=sft_lora_cfg['lora_alpha'],
        lora_dropout=0.0,  # forced to 0 in stacked mode
        target_modules=sft_lora_cfg['target_modules'],
        bias="none",
        task_type=None,
    )

    # GRPO LoRA config (independent of SFT)
    grpo_lora_cfg = model_cfg.get('grpo_lora', {})
    grpo_lora_config = LoraConfig(
        r=grpo_lora_cfg.get('r', 8),
        lora_alpha=grpo_lora_cfg.get('lora_alpha', 16),
        lora_dropout=0.0,  # forced to 0 to avoid sampling/logprob inconsistency
        target_modules=grpo_lora_cfg.get('target_modules',
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"]),
        bias="none",
        task_type=None,
    )

    llm_model = model.backbone.model

    # Determine SFT LoRA path type
    is_peft_dir = (
        sft_lora_path
        and os.path.isdir(sft_lora_path)
        and os.path.exists(os.path.join(sft_lora_path, 'adapter_config.json'))
    )
    is_pt_with_lora = False
    ckpt = None
    if sft_lora_path and not is_peft_dir and os.path.isfile(sft_lora_path) and sft_lora_path.endswith('.pt'):
        ckpt = torch.load(sft_lora_path, map_location='cpu')
        if 'lora' in ckpt and ckpt['lora']:
            is_pt_with_lora = True

    if is_peft_dir:
        # ====== Scenario B: PEFT directory -> load as sft adapter (no merge) ======
        llm_model.model = PM.from_pretrained(
            llm_model.model, sft_lora_path,
            adapter_name=SFT_ADAPTER, is_trainable=False,
        )
        # Freeze sft adapter
        for n, p in llm_model.model.named_parameters():
            if 'lora_' in n and f'.{SFT_ADAPTER}.' in n:
                p.requires_grad_(False)

        # Stack grpo adapter
        llm_model.model.add_adapter(GRPO_ADAPTER, grpo_lora_config)

        model.backbone.model = llm_model
        model.backbone._use_lora = True
        # Set Policy mode: dual LoRA active
        model.backbone.set_active_adapters([SFT_ADAPTER, GRPO_ADAPTER])

        if rank == 0:
            logger.info("✓ Scenario B: sft adapter (frozen, PEFT dir) + grpo adapter (trainable)")
            logger.info(f"  Policy  = base + {SFT_ADAPTER}(r={sft_lora_cfg['r']}) + {GRPO_ADAPTER}(r={grpo_lora_cfg.get('r', 8)})")
            logger.info(f"  Reference = base + {SFT_ADAPTER}")
        return False

    elif is_pt_with_lora:
        # ====== Scenario C: .pt file -> load SFT weights into sft adapter ======
        peft_model = llm_model.model  # build_mllm_backbone already created default adapter

        if not isinstance(peft_model, PM):
            raise RuntimeError("Scenario C: build_mllm_backbone(use_lora=True) should have created PeftModel")

        # 1. Load SFT LoRA weights into default adapter
        lora_state = ckpt['lora']
        loaded = 0
        for n, p in model.backbone.named_parameters():
            if n in lora_state:
                p.data.copy_(lora_state[n])
                loaded += 1
        if loaded == 0:
            for n, p in model.backbone.named_parameters():
                short_key = n[len('model.'):] if n.startswith('model.') else n
                if short_key in lora_state:
                    p.data.copy_(lora_state[short_key])
                    loaded += 1
        if rank == 0:
            logger.info(f"✓ SFT LoRA loaded into default adapter ({loaded} params)")

        # 2. Create sft adapter and copy default weights
        peft_model.add_adapter(SFT_ADAPTER, sft_lora_config)
        default_state = {}
        for n, p in peft_model.named_parameters():
            if 'lora_' in n and '.default.' in n:
                default_state[n] = p.data.clone()
        for n, p in peft_model.named_parameters():
            if 'lora_' in n and f'.{SFT_ADAPTER}.' in n:
                default_name = n.replace(f'.{SFT_ADAPTER}.', '.default.')
                if default_name in default_state:
                    p.data.copy_(default_state[default_name])
                p.requires_grad_(False)
        if rank == 0:
            logger.info("✓ sft adapter created and frozen (weights copied from default)")

        # 3. Delete default adapter
        peft_model.delete_adapter('default')
        if rank == 0:
            logger.info("✓ default adapter deleted")

        # 4. Create grpo adapter
        peft_model.add_adapter(GRPO_ADAPTER, grpo_lora_config)

        model.backbone.model = llm_model
        model.backbone._use_lora = True
        # Set Policy mode
        model.backbone.set_active_adapters([SFT_ADAPTER, GRPO_ADAPTER])

        if rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_grpo = sum(p.numel() for n, p in model.named_parameters()
                                if p.requires_grad and 'lora_' in n and GRPO_ADAPTER in n)
            logger.info(f"[Param stats] Total params: {total_params:,}")
            logger.info(f"[Param stats] Trainable GRPO LoRA params: {trainable_grpo:,}")
            logger.info(f"✓ Scenario C: {SFT_ADAPTER} adapter (frozen) + {GRPO_ADAPTER} adapter (trainable)")
            logger.info(f"  Policy    = base + {SFT_ADAPTER}(r={sft_lora_cfg['r']}) + {GRPO_ADAPTER}(r={grpo_lora_cfg.get('r', 8)})")
            logger.info(f"  Reference = base + {SFT_ADAPTER}")
            logger.info("  Base (4th layer) = disable all adapters")
        return False

    else:
        # ====== Scenario A: No SFT LoRA -> only create grpo adapter ======
        if isinstance(llm_model.model, PM):
            # backbone may have already created default adapter
            if 'default' in llm_model.model.peft_config:
                llm_model.model.delete_adapter('default')
            llm_model.model.add_adapter(GRPO_ADAPTER, grpo_lora_config)
        else:
            llm_model.model = get_peft_model(llm_model.model, grpo_lora_config, adapter_name=GRPO_ADAPTER)

        model.backbone.model = llm_model
        model.backbone._use_lora = True
        model.backbone.set_active_adapters([GRPO_ADAPTER])

        if rank == 0:
            logger.info(f"✓ Scenario A: {GRPO_ADAPTER} adapter (trainable, randomly initialized)")
            logger.info(f"  Policy    = base + {GRPO_ADAPTER}")
            logger.info("  Reference = base (disable all)")
        return True  # Reference = pure base


def compute_rule_reward(generated_text: str, hop_count: int = None) -> float:
    """
    v6-style rule reward: format 1/3 + step count 1/3 + modality tags 1/3
    Returns [0, 1]
    """
    score = 0.0

    # Format: has Step N: and no extra output
    has_format = bool(re.search(r'Step\s+\d+:', generated_text, re.IGNORECASE))
    if has_format:
        # Simplified extra output check
        step_pattern = r'(Step\s+\d+:\s*.*?)(?=Step\s+\d+:|$)'
        matches = re.findall(step_pattern, generated_text, re.IGNORECASE | re.DOTALL)
        reconstructed_len = sum(len(m.strip()) for m in matches) if matches else 0
        original_len = len(generated_text.strip())
        has_extra = original_len > reconstructed_len * 1.1 if reconstructed_len > 0 else True
        if not has_extra:
            score += 1/3
        else:
            score += 1/6

    # Step count (if hop_count reference is available)
    if hop_count is not None:
        actual_steps = len(re.findall(r'Step\s+\d+:', generated_text, re.IGNORECASE))
        if actual_steps == hop_count:
            score += 1/3
    else:
        # No hop_count: check if there are at least 2 steps
        actual_steps = len(re.findall(r'Step\s+\d+:', generated_text, re.IGNORECASE))
        if actual_steps >= 2:
            score += 1/3

    # Modality tags: both [Image] and [Text] present
    steps = re.findall(
        r'Step\s+\d+:\s*\[(\w+)\]', generated_text, re.IGNORECASE
    )
    modalities = {s.lower() for s in steps}
    if 'image' in modalities and 'text' in modalities:
        score += 1/3

    return score


def compute_question_similarity(
    generated_text: str,
    gt_chain: str,
    answer: str,
    question_type: str,
    reranker_model,
    reranker_tokenizer,
    yes_id: int,
    no_id: int,
) -> float:
    """
    Compute semantic similarity between generated question and GT question using Qwen3-Reranker.
    Returns [0, 1]
    """
    # Check if [Question] exists and is after Steps
    if '[Question]' not in generated_text:
        return 0.0

    step_matches = list(re.finditer(r'Step\s+\d+:', generated_text, re.IGNORECASE))
    if step_matches:
        last_step_pos = step_matches[-1].end()
        question_pos = generated_text.find('[Question]')
        if question_pos < last_step_pos:
            return 0.0

    # Extract generated question
    match = re.search(r'\[Question\]\s*(.+?)$', generated_text, re.IGNORECASE | re.DOTALL)
    generated_question = match.group(1).strip() if match else ""
    if not generated_question:
        return 0.0

    # Extract GT question
    if isinstance(gt_chain, list):
        gt_chain = " ".join(gt_chain)
    gt_match = re.search(r'\[Question\]\s*(.+?)$', gt_chain, re.IGNORECASE | re.DOTALL)
    gt_question = gt_match.group(1).strip() if gt_match else ""
    if not gt_question:
        return 0.0

    # Reranker scoring
    context_parts = []
    if answer:
        context_parts.append(f"answer: {answer}")
    if question_type:
        context_parts.append(f"question type: {question_type}")
    context = ", ".join(context_parts)

    if context:
        instruction = (
            f"Given that {context}, "
            "judge if the two questions are semantically equivalent and asking about the same thing."
        )
    else:
        instruction = "Judge if the two questions are semantically equivalent and asking about the same thing."

    prompt = (
        "<|im_start|>system\n"
        'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
        'Note that the answer can only be "yes" or "no".\n'
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"<Instruct>: {instruction}\n"
        f"<Query>: {generated_question}\n"
        f"<Document>: {gt_question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )

    inputs = reranker_tokenizer(
        prompt, return_tensors='pt', truncation=True, max_length=512
    ).to(reranker_model.device)
    with torch.no_grad():
        logits = reranker_model(**inputs).logits[0, -1, :]
        scores = torch.softmax(logits[[no_id, yes_id]].float(), dim=0)
    return scores[1].item()


def batch_compute_question_similarity(
    generated_texts: list,
    gt_chains: list,
    answers: list,
    question_types: list,
    reranker_model,
    reranker_tokenizer,
    yes_id: int,
    no_id: int,
) -> torch.Tensor:
    """Batch reranker similarity, returns [B] tensor on reranker_model.device"""
    prompts = []
    valid_mask = []

    for i in range(len(generated_texts)):
        generated_text = generated_texts[i]
        gt_chain = gt_chains[i]

        if '[Question]' not in generated_text:
            prompts.append("")
            valid_mask.append(False)
            continue

        step_matches = list(re.finditer(r'Step\s+\d+:', generated_text, re.IGNORECASE))
        if step_matches:
            last_step_pos = step_matches[-1].end()
            question_pos = generated_text.find('[Question]')
            if question_pos < last_step_pos:
                prompts.append("")
                valid_mask.append(False)
                continue

        match = re.search(r'\[Question\]\s*(.+?)$', generated_text, re.IGNORECASE | re.DOTALL)
        generated_question = match.group(1).strip() if match else ""
        if not generated_question:
            prompts.append("")
            valid_mask.append(False)
            continue

        if isinstance(gt_chain, list):
            gt_chain = " ".join(gt_chain)
        gt_match = re.search(r'\[Question\]\s*(.+?)$', gt_chain, re.IGNORECASE | re.DOTALL)
        gt_question = gt_match.group(1).strip() if gt_match else ""
        if not gt_question:
            prompts.append("")
            valid_mask.append(False)
            continue

        context_parts = []
        if answers[i]:
            context_parts.append(f"answer: {answers[i]}")
        if question_types[i]:
            context_parts.append(f"question type: {question_types[i]}")
        context = ", ".join(context_parts)

        if context:
            instruction = (
                f"Given that {context}, "
                "judge if the two questions are semantically equivalent and asking about the same thing."
            )
        else:
            instruction = "Judge if the two questions are semantically equivalent and asking about the same thing."

        prompt = (
            "<|im_start|>system\n"
            'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
            'Note that the answer can only be "yes" or "no".\n'
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"<Instruct>: {instruction}\n"
            f"<Query>: {generated_question}\n"
            f"<Document>: {gt_question}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )
        prompts.append(prompt)
        valid_mask.append(True)

    scores = torch.zeros(len(generated_texts), device=reranker_model.device)
    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    if valid_indices:
        valid_prompts = [prompts[i] for i in valid_indices]
        inputs = reranker_tokenizer(
            valid_prompts, return_tensors='pt', truncation=True,
            max_length=512, padding=True,
        ).to(reranker_model.device)
        with torch.no_grad():
            logits = reranker_model(**inputs).logits[:, -1, :]
            batch_scores = torch.softmax(logits[:, [no_id, yes_id]].float(), dim=-1)[:, 1]
        for idx, vi in enumerate(valid_indices):
            scores[vi] = batch_scores[idx]

    return scores


def batch_compute_question_similarity_no_cot(
    generated_texts: list,
    gt_questions: list,
    answers: list,
    question_types: list,
    reranker_model,
    reranker_tokenizer,
    yes_id: int,
    no_id: int,
) -> torch.Tensor:
    """no-CoT batch reranker similarity: entire generated text is the question, GT from batch question field"""
    prompts = []
    valid_mask = []

    for i in range(len(generated_texts)):
        generated_question = generated_texts[i].strip()
        gt_question = gt_questions[i].strip() if gt_questions[i] else ""

        if not generated_question or not gt_question:
            prompts.append("")
            valid_mask.append(False)
            continue

        context_parts = []
        if answers[i]:
            context_parts.append(f"answer: {answers[i]}")
        if question_types[i]:
            context_parts.append(f"question type: {question_types[i]}")
        context = ", ".join(context_parts)

        if context:
            instruction = (
                f"Given that {context}, "
                "judge if the two questions are semantically equivalent and asking about the same thing."
            )
        else:
            instruction = "Judge if the two questions are semantically equivalent and asking about the same thing."

        prompt = (
            "<|im_start|>system\n"
            'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
            'Note that the answer can only be "yes" or "no".\n'
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"<Instruct>: {instruction}\n"
            f"<Query>: {generated_question}\n"
            f"<Document>: {gt_question}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )
        prompts.append(prompt)
        valid_mask.append(True)

    scores = torch.zeros(len(generated_texts), device=reranker_model.device)
    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    if valid_indices:
        valid_prompts = [prompts[i] for i in valid_indices]
        inputs = reranker_tokenizer(
            valid_prompts, return_tensors='pt', truncation=True,
            max_length=512, padding=True,
        ).to(reranker_model.device)
        with torch.no_grad():
            logits = reranker_model(**inputs).logits[:, -1, :]
            batch_scores = torch.softmax(logits[:, [no_id, yes_id]].float(), dim=-1)[:, 1]
        for idx, vi in enumerate(valid_indices):
            scores[vi] = batch_scores[idx]

    return scores


def train_grpo(config_path: str, deepspeed_config: str):
    rank, world_size, local_rank = setup_distributed()

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    set_seed(config.get('seed', 42))

    if rank == 0:
        logger.info("=" * 70)
        logger.info("MLLM GaussMI GRPO Training (Stacked LoRA + PID Reward)")
        logger.info(f"World size: {world_size}, Rank: {rank}")
        logger.info("=" * 70)

    # ====== Create model ======
    model_cfg = config['model']
    backbone_type = model_cfg['backbone_type']

    # Check if SFT LoRA checkpoint (.pt file) exists
    sft_lora_path = config['training'].get('load_pretrained_lora')
    has_sft_lora_pt = (
        sft_lora_path
        and os.path.isfile(sft_lora_path)
        and sft_lora_path.endswith('.pt')
        and 'lora' in torch.load(sft_lora_path, map_location='cpu')
    )

    # Get SFT lora config (use defaults if not in config file)
    sft_lora_config = model_cfg.get('lora')
    if has_sft_lora_pt and not sft_lora_config:
        # Use default config consistent with SFT training
        sft_lora_config = {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            'bias': "none",
        }
        if rank == 0:
            logger.info("Using default SFT LoRA config")

    if rank == 0:
        logger.info(f"[DEBUG] has_sft_lora_pt: {has_sft_lora_pt}")
        logger.info(f"[DEBUG] sft_lora_path: {sft_lora_path}")
        logger.info(f"[DEBUG] sft_lora_config: {sft_lora_config}")

    # If SFT LoRA (.pt) exists, enable LoRA when building backbone
    backbone = build_mllm_backbone(
        backbone_type=backbone_type,
        model_path=model_cfg['model_path'],
        use_flash_attention=model_cfg.get('use_flash_attention', False),
        use_gradient_checkpointing=model_cfg.get('use_gradient_checkpointing', False),
        use_lora=has_sft_lora_pt,
        lora_config=sft_lora_config if has_sft_lora_pt else None,
        freeze_visual_encoder=model_cfg.get('freeze_visual_encoder', True),
    )

    dis_cfg = model_cfg.get('disentangle', {})
    model = MLLMEarlyDisentangleGaussMIModel(backbone=backbone, disentangle_config=dis_cfg)

    # Load SFT disentangle modules
    load_pretrained_checkpoint(model, config, rank)

    # Stack LoRA
    use_base_as_ref = setup_stacked_lora(model, config, rank)
    policy_adapters = [GRPO_ADAPTER] if use_base_as_ref else [SFT_ADAPTER, GRPO_ADAPTER]

    # Freeze
    configure_freezing(model, config, rank)

    # setup_stacked_lora already set grad via set_active_adapters -> _enforce_grad_policy
    # configure_freezing may reset backbone non-LoRA params, re-ensure dual LoRA state here
    model.backbone.set_active_adapters(policy_adapters)

    if rank == 0:
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Final trainable params: {num_trainable:,}")

    # ====== Load PID Scorer ======
    grpo_cfg = config['grpo']
    pid_mode = grpo_cfg.get('pid_mode', 'vmf')
    pid_scorer_path = grpo_cfg['pid_scorer_path']
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if pid_mode == 'vmf':
        pid_wrapper = vMFScorerWrapper.from_checkpoint(
            scorer_path=pid_scorer_path, device=device
        )
    elif pid_mode == 'gaussian':
        pid_projector_path = grpo_cfg['pid_projector_path']
        pid_wrapper = PIDScorerWrapper.from_checkpoint(
            projector_path=pid_projector_path,
            scorer_path=pid_scorer_path,
            device=device
        )
    else:
        raise ValueError(f"Unknown pid_mode: {pid_mode}, supported: 'vmf' or 'gaussian'")

    if rank == 0:
        logger.info(f"✓ PID Scorer loaded (mode={pid_mode}): {pid_scorer_path}")

    # ====== Load Qwen3-Reranker (question similarity) ======
    reranker_path = grpo_cfg.get('reranker_model_path', '../../models/Qwen3-Reranker-4B')
    reranker_tokenizer = HFAutoTokenizer.from_pretrained(reranker_path)
    reranker_model = AutoModelForCausalLM.from_pretrained(
        reranker_path, torch_dtype=torch.float16
    ).to(device).eval()
    reranker_yes_id = reranker_tokenizer("yes", add_special_tokens=False)["input_ids"][0]
    reranker_no_id = reranker_tokenizer("no", add_special_tokens=False)["input_ids"][0]
    if rank == 0:
        logger.info(f"✓ Qwen3-Reranker loaded: {reranker_path}")

    # ====== Dataset ======
    processor = backbone.get_processor()
    tokenizer = backbone.get_tokenizer()
    data_cfg = config['data']
    no_cot = config.get('no_cot', False)
    DatasetClass = MLLMNoCotDataset if no_cot else MLLMDataset
    if rank == 0 and no_cot:
        logger.info("no_cot=True, using MLLMNoCotDataset (question-only generation, skip reasoning chain)")

    collate = partial(
        mllm_collate_fn, backbone_type=backbone_type,
        pad_token_id=tokenizer.pad_token_id or 0,
    )

    train_dataset = DatasetClass(
        data_file=data_cfg['train_file'],
        image_root=data_cfg['image_root'],
        processor=processor, tokenizer=tokenizer,
        backbone_type=backbone_type,
        max_constraint_length=data_cfg.get('max_constraint_length', 256),
        max_target_length=data_cfg.get('max_target_length', 512),
        filter_non_multimodal=data_cfg.get('filter_non_multimodal', True),
        preprocess_workers=data_cfg.get('preprocess_workers', 4),
    )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    ) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4, collate_fn=collate, pin_memory=True,
    )

    if rank == 0:
        logger.info(f"Training set: {len(train_dataset)} samples")

    # ====== Validation set ======
    val_dataset = DatasetClass(
        data_file=data_cfg['val_file'],
        image_root=data_cfg['image_root'],
        processor=processor, tokenizer=tokenizer,
        backbone_type=backbone_type,
        max_constraint_length=data_cfg.get('max_constraint_length', 256),
        max_target_length=data_cfg.get('max_target_length', 512),
        filter_non_multimodal=data_cfg.get('filter_non_multimodal', True),
        preprocess_workers=data_cfg.get('preprocess_workers', 4),
    )

    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    ) if world_size > 1 else None

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=4, collate_fn=collate, pin_memory=True,
    )

    if rank == 0:
        logger.info(f"Validation set: {len(val_dataset)} samples")

    # ====== Optimizer param groups ======
    grpo_lora_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and 'lora' in n.lower() and GRPO_ADAPTER in n
    ]
    param_groups = [{
        'params': grpo_lora_params,
        'lr': config['training']['learning_rate']['grpo_lora'],
        'name': 'grpo_lora',
    }]

    if rank == 0:
        logger.info(f"Optimizer: {len(grpo_lora_params)} params, lr={param_groups[0]['lr']:.2e}")

    # ====== DeepSpeed ======
    with open(deepspeed_config, 'r') as f:
        ds_config = json.load(f)

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, model_parameters=param_groups, config=ds_config,
    )

    # ====== Training hyperparams ======
    grpo_cfg = config['grpo']
    num_samples = grpo_cfg['num_samples']
    clip_eps = grpo_cfg['clip_eps']
    temperature = grpo_cfg['temperature']
    beta = grpo_cfg.get('beta', 0.1)
    use_kl_penalty = grpo_cfg.get('use_kl_penalty', True)
    max_new_tokens = config['inference']['max_new_tokens']
    rule_weight = grpo_cfg['reward']['rule_weight']
    sim_weight = grpo_cfg['reward']['sim_weight']
    pid_weight = grpo_cfg['reward']['pid_weight']
    ppo_cot_weight = grpo_cfg.get('ppo_cot_weight', 0.3)
    ppo_q_weight = grpo_cfg.get('ppo_q_weight', 1.0)
    kl_cot_weight = grpo_cfg.get('kl_cot_weight', 1.1)
    kl_q_weight = grpo_cfg.get('kl_q_weight', 1.0)

    # ====== Output directory ======
    output_dir = Path(config['output']['save_dir'])
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir = Path(config['output']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
    else:
        writer = None

    # === PLACEHOLDER_TRAIN_LOOP ===

    num_epochs = config['training']['num_epochs']
    best_avg_reward = -float('inf')
    global_step = 0
    pad_id = tokenizer.pad_token_id or 0
    G = num_samples

    # Ensure Policy mode at training start
    backbone = model_engine.module.backbone
    backbone.set_active_adapters(policy_adapters)
    if rank == 0:
        peft_llm = backbone.model.model
        active = getattr(peft_llm.base_model, 'active_adapters', None) or getattr(peft_llm.base_model, 'active_adapter', 'unknown')
        logger.info(f"Training started, active adapters: {active}")
        for n, p in peft_llm.named_parameters():
            if 'lora_' in n and p.requires_grad:
                logger.info(f"  trainable: {n}")
                break  # only print first to avoid flooding logs

    for epoch in range(num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

        model_engine.train()
        epoch_loss = 0.0
        epoch_rewards = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}") if rank == 0 else train_loader

        for batch_idx, batch in enumerate(pbar):
            device = model_engine.device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            batch.pop('id', None)
            answers = batch.pop('answer', [''] * batch['pixel_values'].size(0))
            question_types = batch.pop('question_type', [''] * batch['pixel_values'].size(0))
            reasoning_chains = batch.pop('reasoning_chain', [''] * batch['pixel_values'].size(0))
            gt_questions = batch.pop('question', [''] * batch['pixel_values'].size(0))

            encode_kwargs = {}
            if 'image_grid_thw' in batch:
                encode_kwargs['image_grid_thw'] = batch.pop('image_grid_thw')

            batch_size = batch['text_input_ids'].size(0)

            # ====== 0. Visual encoding (one-shot, cached for reuse) ======
            model_engine.eval()
            with torch.no_grad():
                soft_prompt, soft_mask, dis_out = model_engine.module._encode_and_disentangle(
                    pixel_values=batch['pixel_values'],
                    num_images=batch.get('num_images'),
                    text_input_ids=batch['text_input_ids'],
                    text_attention_mask=batch['text_attention_mask'],
                    encode_kwargs=encode_kwargs,
                )
                cached_dis_out = {k: v.detach() for k, v in dis_out.items()}
                soft_prompt = soft_prompt.detach()
                soft_mask = soft_mask.detach()

            # ====== 1. Sampling phase (batched parallel: G split in halves) ======
            all_generated_ids = []
            all_current_log_probs = []
            all_current_attn_masks = []
            all_generated_texts = []

            # Split into two batches: [0, G/2) and [G/2, G)
            G_half = G // 2
            batches = [(0, G_half), (G_half, G)]

            with torch.no_grad():
                for start_idx, end_idx in batches:
                    n_samples = end_idx - start_idx
                    # Expand soft_prompt [B, N, H] -> [B*n_samples, N, H]
                    exp_soft_prompt = soft_prompt.repeat_interleave(n_samples, dim=0)
                    exp_soft_mask = soft_mask.repeat_interleave(n_samples, dim=0)
                    exp_sys_ids = batch['system_input_ids'].repeat_interleave(n_samples, dim=0)
                    exp_sys_mask = batch['system_attention_mask'].repeat_interleave(n_samples, dim=0)
                    exp_con_ids = batch['constraint_input_ids'].repeat_interleave(n_samples, dim=0)
                    exp_con_mask = batch['constraint_attention_mask'].repeat_interleave(n_samples, dim=0)

                    result = model_engine.module.generate_from_cached_prompt(
                        soft_prompt=exp_soft_prompt,
                        soft_mask=exp_soft_mask,
                        system_input_ids=exp_sys_ids,
                        system_attention_mask=exp_sys_mask,
                        constraint_input_ids=exp_con_ids,
                        constraint_attention_mask=exp_con_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                    )

                    # Reshape output: [B*n_samples, gen_len] -> n_samples x [B, gen_len]
                    max_gen = result['generated_ids'].size(1)
                    gen_ids_bg = result['generated_ids'].view(batch_size, n_samples, max_gen)
                    lp_bg = result['token_log_probs'].view(batch_size, n_samples, max_gen)
                    am_bg = result['attention_mask'].view(batch_size, n_samples, max_gen)

                    for s in range(n_samples):
                        all_generated_ids.append(gen_ids_bg[:, s, :].contiguous().detach())
                        all_current_log_probs.append(lp_bg[:, s, :].contiguous().detach())
                        all_current_attn_masks.append(am_bg[:, s, :].contiguous().detach())

                    # generated_texts order: B0G0, B0G1, ..., B(B-1)G(n-1)
                    _texts = result['generated_texts']
                    for s in range(n_samples):
                        all_generated_texts.append([_texts[b * n_samples + s] for b in range(batch_size)])

                    del exp_soft_prompt, exp_soft_mask, exp_sys_ids, exp_sys_mask, exp_con_ids, exp_con_mask

            # ====== 2. Reward computation (three-way: rule + q_similarity + PID) ======
            rewards_list = []
            total_rule_reward = 0.0
            total_sim_reward = 0.0
            total_pid_reward = 0.0

            with torch.no_grad():
                for s_idx in range(G):
                    if no_cot:
                        # no-CoT: entire generated text is the question, use generated_ids directly
                        q_ids = all_generated_ids[s_idx]
                        q_mask = (q_ids != pad_id).long()
                    else:
                        q_ids = extract_question_ids(
                            all_generated_ids[s_idx], tokenizer, pad_id
                        )
                        q_mask = (q_ids != pad_id).long()

                    q_hidden = model_engine.module.backbone.encode_text(
                        q_ids, q_mask,
                        model_engine.module.n_text_layers,
                    )

                    pid_scores = pid_wrapper.compute_reward(
                        cached_dis_out, q_hidden, q_mask
                    )

                    if no_cot:
                        rule_scores = torch.zeros(batch_size, device=device)
                        sim_scores = batch_compute_question_similarity_no_cot(
                            all_generated_texts[s_idx],
                            gt_questions,
                            answers,
                            question_types,
                            reranker_model,
                            reranker_tokenizer,
                            reranker_yes_id,
                            reranker_no_id,
                        )
                    else:
                        rule_scores = torch.tensor([
                            compute_rule_reward(all_generated_texts[s_idx][b])
                            for b in range(batch_size)
                        ], device=device, dtype=torch.float32)

                        sim_scores = batch_compute_question_similarity(
                            all_generated_texts[s_idx],
                            reasoning_chains,
                            answers,
                            question_types,
                            reranker_model,
                            reranker_tokenizer,
                            reranker_yes_id,
                            reranker_no_id,
                        )

                    sample_rewards = (
                        rule_weight * rule_scores
                        + sim_weight * sim_scores
                        + pid_weight * pid_scores
                    )
                    rewards_list.append(sample_rewards)

                    # Accumulate per-component means (for monitoring)
                    total_rule_reward += rule_scores.mean().item()
                    total_sim_reward += sim_scores.mean().item()
                    total_pid_reward += pid_scores.mean().item()

            # Per-component average rewards
            avg_rule_reward = total_rule_reward / G
            avg_sim_reward = total_sim_reward / G
            avg_pid_reward = total_pid_reward / G

            rewards = torch.stack(rewards_list, dim=1)  # [B, G]
            epoch_rewards.append(rewards.mean().item())

            # ====== 3. Reference log probs (batched parallel: G split in halves) ======
            all_ref_log_probs = []
            with torch.no_grad(), backbone.reference_mode(use_base_as_ref):

                for start_idx, end_idx in batches:
                        n_samples = end_idx - start_idx
                        # Expand soft_prompt
                        exp_soft_prompt = soft_prompt.repeat_interleave(n_samples, dim=0)
                        exp_soft_mask = soft_mask.repeat_interleave(n_samples, dim=0)
                        exp_sys_ids = batch['system_input_ids'].repeat_interleave(n_samples, dim=0)
                        exp_sys_mask = batch['system_attention_mask'].repeat_interleave(n_samples, dim=0)
                        exp_con_ids = batch['constraint_input_ids'].repeat_interleave(n_samples, dim=0)
                        exp_con_mask = batch['constraint_attention_mask'].repeat_interleave(n_samples, dim=0)

                        # Align generated_ids lengths for this batch
                        batch_gids = all_generated_ids[start_idx:end_idx]
                        max_gen_all = max(gid.size(1) for gid in batch_gids)
                        padded_gids = []
                        for gid in batch_gids:
                            pad_len = max_gen_all - gid.size(1)
                            if pad_len > 0:
                                gid = F.pad(gid, (0, pad_len), value=pad_id)
                            padded_gids.append(gid)
                        exp_gen_ids = torch.stack(padded_gids, dim=1).reshape(batch_size * n_samples, max_gen_all)

                        ref_result = model_engine.module.compute_log_probs_from_cached_prompt(
                            soft_prompt=exp_soft_prompt,
                            soft_mask=exp_soft_mask,
                            system_input_ids=exp_sys_ids,
                            system_attention_mask=exp_sys_mask,
                            constraint_input_ids=exp_con_ids,
                            constraint_attention_mask=exp_con_mask,
                            generated_ids=exp_gen_ids,
                        )

                        # Reshape: [B*n_samples, max_gen_all] -> n_samples x [B, max_gen_all]
                        ref_lp_bg = ref_result['token_log_probs'].view(batch_size, n_samples, max_gen_all)
                        for s in range(n_samples):
                            all_ref_log_probs.append(ref_lp_bg[:, s, :].contiguous().detach())

                        del exp_soft_prompt, exp_soft_mask, exp_sys_ids, exp_sys_mask, exp_con_ids, exp_con_mask, exp_gen_ids

            # ====== 4. Advantages ======
            advantages = compute_advantages(rewards)

            # ====== 5. PPO clip loss (reusing soft_prompt) ======
            model_engine.train()
            for m in model_engine.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.training = False

            total_loss_acc = 0.0
            total_entropy_acc = 0.0
            total_entropy_cot_acc = 0.0
            total_entropy_q_acc = 0.0
            total_clip_cot_acc = 0.0
            total_clip_q_acc = 0.0
            total_kl_cot_acc = 0.0
            total_kl_q_acc = 0.0
            total_q_token_frac_acc = 0.0
            total_clip_loss_acc = 0.0
            total_kl_loss_acc = 0.0

            for s_idx in range(G):
                result = model_engine.module.compute_log_probs_from_cached_prompt(
                    soft_prompt=soft_prompt,
                    soft_mask=soft_mask,
                    system_input_ids=batch['system_input_ids'],
                    system_attention_mask=batch['system_attention_mask'],
                    constraint_input_ids=batch['constraint_input_ids'],
                    constraint_attention_mask=batch['constraint_attention_mask'],
                    generated_ids=all_generated_ids[s_idx],
                )

                cur_lp = result['token_log_probs']
                attn_out = result['attention_mask']
                token_entropy = result['token_entropy']
                gen_len = cur_lp.size(1)

                # Align behavior log probs length
                beh_lp = all_current_log_probs[s_idx]
                if beh_lp.size(1) < gen_len:
                    beh_lp = F.pad(beh_lp, (0, gen_len - beh_lp.size(1)))
                elif beh_lp.size(1) > gen_len:
                    beh_lp = beh_lp[:, :gen_len]

                ref_lp = all_ref_log_probs[s_idx]
                if ref_lp.size(1) < gen_len:
                    ref_lp = F.pad(ref_lp, (0, gen_len - ref_lp.size(1)))
                elif ref_lp.size(1) > gen_len:
                    ref_lp = ref_lp[:, :gen_len]

                # Token-level ratio
                ratio = torch.exp(cur_lp - beh_lp)
                adv = advantages[:, s_idx].unsqueeze(1)

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
                clip_loss_token = -torch.min(surr1, surr2)

                attn_f = attn_out.float()
                mean_entropy = (token_entropy * attn_f).sum(dim=1) / attn_f.sum(dim=1).clamp(min=1)
                total_entropy_acc += mean_entropy.mean().item()

                if no_cot:
                    # Unified loss: no segmentation, compute over entire attention mask range
                    clip_loss = (clip_loss_token * attn_f).sum(dim=1) / attn_f.sum(dim=1).clamp(min=1)
                    clip_loss = clip_loss.mean()
                    total_clip_loss_acc += clip_loss.item()

                    kl_loss = torch.tensor(0.0, device=device)
                    if use_kl_penalty and beta > 0:
                        log_ratio = cur_lp - ref_lp
                        ratio_k3 = torch.exp(log_ratio.clamp(-10, 10))
                        token_kl = ratio_k3 - log_ratio - 1
                        kl_loss = beta * (token_kl * attn_f).sum(dim=1) / attn_f.sum(dim=1).clamp(min=1)
                        kl_loss = kl_loss.mean()
                        total_kl_loss_acc += kl_loss.item()
                else:
                    # Segmented mask
                    with torch.no_grad():
                        _, q_starts = extract_question_ids(
                            all_generated_ids[s_idx], tokenizer, pad_id,
                            return_start_positions=True,
                        )

                        eos_id = tokenizer.eos_token_id
                        eos_positions = (all_generated_ids[s_idx] == eos_id).long()

                        positions = torch.arange(gen_len, device=device).unsqueeze(0)
                        if eos_positions.any():
                            eos_cumsum = torch.cumsum(eos_positions, dim=1)
                            eos_mask = (eos_cumsum > 0).float()
                        else:
                            eos_mask = torch.zeros_like(positions).float()

                        q_mask_seg = (positions >= q_starts.unsqueeze(1)).float()

                    cot_mask = attn_f * (1 - q_mask_seg) * (1 - eos_mask)
                    q_mask_f = attn_f * q_mask_seg * (1 - eos_mask)

                    cot_entropy = (token_entropy * cot_mask).sum(dim=1) / cot_mask.sum(dim=1).clamp(min=1)
                    q_entropy = (token_entropy * q_mask_f).sum(dim=1) / q_mask_f.sum(dim=1).clamp(min=1)
                    total_entropy_cot_acc += cot_entropy.mean().item()
                    total_entropy_q_acc += q_entropy.mean().item()

                    total_q_token_frac_acc += (q_mask_f.sum() / attn_f.sum().clamp(min=1)).item()

                    clip_loss_cot = (clip_loss_token * cot_mask).sum(dim=1) / cot_mask.sum(dim=1).clamp(min=1)
                    clip_loss_q = (clip_loss_token * q_mask_f).sum(dim=1) / q_mask_f.sum(dim=1).clamp(min=1)
                    clip_loss = (ppo_cot_weight * clip_loss_cot + ppo_q_weight * clip_loss_q).mean()
                    total_clip_cot_acc += clip_loss_cot.mean().item()
                    total_clip_q_acc += clip_loss_q.mean().item()

                    kl_loss = torch.tensor(0.0, device=device)
                    if use_kl_penalty and beta > 0:
                        log_ratio = cur_lp - ref_lp
                        ratio_k3 = torch.exp(log_ratio.clamp(-10, 10))
                        token_kl = ratio_k3 - log_ratio - 1
                        kl_cot = (token_kl * cot_mask).sum(dim=1) / cot_mask.sum(dim=1).clamp(min=1)
                        kl_q = (token_kl * q_mask_f).sum(dim=1) / q_mask_f.sum(dim=1).clamp(min=1)
                        kl_loss = beta * (kl_cot_weight * kl_cot + kl_q_weight * kl_q).mean()
                        total_kl_cot_acc += kl_cot.mean().item()
                        total_kl_q_acc += kl_q.mean().item()

                loss = clip_loss + kl_loss
                model_engine.backward(loss / G)
                total_loss_acc += loss.item()

            model_engine.step()
            grad_norm = model_engine.get_global_grad_norm()
            global_step += 1
            epoch_loss += total_loss_acc / G

            if rank == 0 and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'loss': f'{total_loss_acc/G:.4f}',
                    'reward': f'{rewards.mean().item():.4f}',
                })

            if rank == 0 and writer:
                writer.add_scalar('train/loss', total_loss_acc / G, global_step)
                writer.add_scalar('reward/total', rewards.mean().item(), global_step)
                writer.add_scalar('reward/rule', avg_rule_reward, global_step)
                writer.add_scalar('reward/sim', avg_sim_reward, global_step)
                writer.add_scalar('reward/pid', avg_pid_reward, global_step)
                writer.add_scalar('train/reward_std', rewards.std().item(), global_step)
                writer.add_scalar('train/policy_entropy', total_entropy_acc / G, global_step)
                if no_cot:
                    writer.add_scalar('train/clip_loss', total_clip_loss_acc / G, global_step)
                    if use_kl_penalty and beta > 0:
                        writer.add_scalar('train/kl_divergence', total_kl_loss_acc / G, global_step)
                else:
                    writer.add_scalar('train/entropy_cot', total_entropy_cot_acc / G, global_step)
                    writer.add_scalar('train/entropy_q', total_entropy_q_acc / G, global_step)
                    if use_kl_penalty and beta > 0:
                        writer.add_scalar('train/kl_divergence', (total_kl_cot_acc + total_kl_q_acc) / G, global_step)
                        writer.add_scalar('train/kl_cot', total_kl_cot_acc / G, global_step)
                        writer.add_scalar('train/kl_q', total_kl_q_acc / G, global_step)
                    writer.add_scalar('train/clip_loss_cot', total_clip_cot_acc / G, global_step)
                    writer.add_scalar('train/clip_loss_q', total_clip_q_acc / G, global_step)
                    writer.add_scalar('train/q_token_frac', total_q_token_frac_acc / G, global_step)
                if grad_norm is not None:
                    writer.add_scalar('train/grad_norm', grad_norm, global_step)
                writer.add_scalar('train/learning_rate', model_engine.optimizer.param_groups[0]['lr'], global_step)

            # Memory cleanup
            if config['training'].get('memory_cleanup_config', {}).get('enable_deep_cleanup', False):
                if (batch_idx + 1) % 10 == 0:
                    deep_memory_cleanup(rank, world_size)

        # Epoch end
        avg_loss = epoch_loss / max(len(train_loader), 1)
        avg_train_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)

        if rank == 0:
            logger.info(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}, avg_train_reward={avg_train_reward:.4f}")

        # ====== Validation ======
        if rank == 0:
            logger.info("Running validation...")
        model_engine.eval()
        # Ensure validation uses Policy adapter
        backbone.set_active_adapters(policy_adapters)
        val_rewards = []
        device = model_engine.device

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation") if rank == 0 else val_loader
            for batch in val_pbar:
                device = model_engine.device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                batch.pop('id', None)
                answers = batch.pop('answer', [''] * batch['text_input_ids'].size(0))
                question_types = batch.pop('question_type', [''] * batch['text_input_ids'].size(0))
                reasoning_chains = batch.pop('reasoning_chain', [''] * batch['text_input_ids'].size(0))
                val_gt_questions = batch.pop('question', [''] * batch['text_input_ids'].size(0))

                encode_kwargs = {}
                if 'image_grid_thw' in batch:
                    encode_kwargs['image_grid_thw'] = batch.pop('image_grid_thw')

                batch_size = batch['text_input_ids'].size(0)

                # Generate once
                result = model_engine.module.generate_with_log_probs(
                    pixel_values=batch['pixel_values'],
                    num_images=batch.get('num_images'),
                    text_input_ids=batch['text_input_ids'],
                    text_attention_mask=batch['text_attention_mask'],
                    system_input_ids=batch['system_input_ids'],
                    system_attention_mask=batch['system_attention_mask'],
                    constraint_input_ids=batch['constraint_input_ids'],
                    constraint_attention_mask=batch['constraint_attention_mask'],
                    encode_kwargs=encode_kwargs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                )

                generated_texts = result['generated_texts']
                generated_ids = result['generated_ids']
                dis_out = result['dis_out']

                # Compute PID scores
                if no_cot:
                    q_ids = generated_ids
                    q_mask = (q_ids != pad_id).long()
                else:
                    q_ids = extract_question_ids(generated_ids, tokenizer, pad_id)
                    q_mask = (q_ids != pad_id).long()
                q_hidden = model_engine.module.backbone.encode_text(
                    q_ids, q_mask, model_engine.module.n_text_layers
                )
                pid_scores = pid_wrapper.compute_reward(dis_out, q_hidden, q_mask)

                # Compute three-way rewards
                for b in range(batch_size):
                    if no_cot:
                        rule_score = 0.0
                        sim_score = batch_compute_question_similarity_no_cot(
                            [generated_texts[b]],
                            [val_gt_questions[b]],
                            [answers[b]],
                            [question_types[b]],
                            reranker_model,
                            reranker_tokenizer,
                            reranker_yes_id,
                            reranker_no_id,
                        ).item()
                    else:
                        rule_score = compute_rule_reward(generated_texts[b])
                        sim_score = compute_question_similarity(
                            generated_texts[b],
                            reasoning_chains[b],
                            answers[b],
                            question_types[b],
                            reranker_model,
                            reranker_tokenizer,
                            reranker_yes_id,
                            reranker_no_id,
                        )
                    reward = (
                        rule_weight * rule_score
                        + sim_weight * sim_score
                        + pid_weight * pid_scores[b].item()
                    )
                    val_rewards.append(reward)

                if rank == 0 and isinstance(val_pbar, tqdm):
                    val_pbar.set_postfix({'reward': f'{sum(val_rewards)/len(val_rewards):.4f}'})

                # Validation cleanup
                if config['training'].get('memory_cleanup_config', {}).get('enable_validation_cleanup', False):
                    deep_memory_cleanup(rank, world_size, sync_barrier=False)

        avg_val_reward = sum(val_rewards) / len(val_rewards) if val_rewards else 0.0

        # Cross-rank sync
        avg_val_tensor = torch.tensor([avg_val_reward], dtype=torch.float32, device=device)
        if world_size > 1:
            torch.distributed.all_reduce(avg_val_tensor, op=torch.distributed.ReduceOp.SUM)
            avg_val_reward = avg_val_tensor.item() / world_size

        if rank == 0:
            logger.info(f"Validation reward: {avg_val_reward:.4f}")
            if writer:
                writer.add_scalar('val/reward', avg_val_reward, epoch + 1)

        # Unified is_best check (avoid NCCL deadlock)
        is_best = avg_val_reward > best_avg_reward
        if world_size > 1:
            is_best_tensor = torch.tensor([int(is_best)], device=device)
            torch.distributed.broadcast(is_best_tensor, src=0)
            is_best = bool(is_best_tensor.item())

        # Save best checkpoint
        if is_best:
            best_avg_reward = avg_val_reward
            if rank == 0:
                logger.info(f"New best validation reward: {best_avg_reward:.4f} - Saving checkpoint...")

            checkpoint_dict = {
                'epoch': epoch + 1,
                'global_step': global_step,
                'val_reward': best_avg_reward,
            }

            # Save disentangle modules (ZeRO-3 compatible)
            for mname in TRAINABLE_MODULES:
                module = getattr(model_engine.module, mname)
                with deepspeed.zero.GatheredParameters(list(module.parameters()), modifier_rank=0):
                    if rank == 0:
                        checkpoint_dict[mname] = module.state_dict()

            # Save LoRA (dual adapter mode: only save grpo adapter)
            lora_state = {}
            for n, p in model_engine.module.backbone.model.named_parameters():
                if 'lora' in n.lower() and GRPO_ADAPTER in n:
                    with deepspeed.zero.GatheredParameters([p], modifier_rank=0):
                        if rank == 0:
                            lora_state[n] = p.data.cpu()
            if rank == 0:
                checkpoint_dict['lora'] = lora_state

            if rank == 0:
                save_path = output_dir / "trainable_best.pt"
                torch.save(checkpoint_dict, save_path)
                logger.info(f"✓ Saved checkpoint to {save_path}")

        if config['training'].get('memory_cleanup_config', {}).get('enable_epoch_cleanup', False):
            deep_memory_cleanup(rank, world_size)

    if rank == 0:
        logger.info(f"Training complete, best avg_reward: {best_avg_reward:.4f}")
        if writer:
            writer.close()

def main():
    parser = argparse.ArgumentParser(description="MLLM GaussMI GRPO Training")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--deepspeed', type=str, required=True)
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()
    train_grpo(args.config, args.deepspeed)


if __name__ == '__main__':
    main()
