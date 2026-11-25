#!/usr/bin/env python
"""
MLLM Early-Style Disentangle Inference Script (Gaussian · CCA-MI + Memory Queue)

Usage:
    Single GPU:
        CUDA_VISIBLE_DEVICES=0 python scripts/inference_mllm_early_disentangle_gauss_mi.py \\
            --config config/mllm_early_disentangle_gauss_mi.yaml \\
            --checkpoint outputs/checkpoints/scienceqa/mllm_early_disentangle_gauss_mi/trainable_best.pt \\
            --test_file data/scienceqa_stage4_sft_caption/test.json \\
            --output_file outputs/predictions/mllm_early_disentangle_gauss_mi.jsonl

    Multi-GPU DDP:
        CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \\
            scripts/inference_mllm_early_disentangle_gauss_mi.py \\
            --config config/mllm_early_disentangle_gauss_mi.yaml \\
            --checkpoint outputs/checkpoints/scienceqa/mllm_early_disentangle_gauss_mi/trainable_best.pt \\
            --test_file data/scienceqa_stage4_sft_caption/test.json \\
            --output_file outputs/predictions/mllm_early_disentangle_gauss_mi.jsonl
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import yaml
import json
import torch
import torch.distributed as dist
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import logging
from pathlib import Path

from src.models.mllm_backbone import build_mllm_backbone, SFT_ADAPTER, GRPO_ADAPTER
from src.models.mllm_early_disentangle_gauss_mi_model import MLLMEarlyDisentangleGaussMIModel
from src.data.mllm_dataset import MLLMDataset, mllm_collate_fn
from src.data.mllm_ablation_no_cot_dataset import MLLMNoCotDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Module names saved in checkpoint, consistent with training script
TRAINABLE_MODULES = [
    'shared_dual',
    'visual_private',
    'text_private',
    'segment_embedding',
]


def extract_question(generated_text: str) -> str:
    """Extract content after the first [Question] tag, truncate at next ["""
    marker = "[Question]"
    if marker not in generated_text:
        return generated_text.strip()

    content = generated_text[generated_text.index(marker) + len(marker):]

    pos = content.find("[")
    if pos != -1:
        content = content[:pos]

    return content.strip()


def extract_question_no_cot(generated_text: str) -> str:
    """no-CoT mode: model outputs question text directly, no tag parsing needed"""
    return generated_text.strip()


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank, world_size, local_rank = 0, 1, 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        dist.barrier()

    return rank, world_size, local_rank


def _get_peft_model(llm_model):
    """Get PeftModel reference based on backbone type"""
    from peft import PeftModel
    for attr in ('model', 'language_model'):
        sub = getattr(llm_model, attr, None)
        if isinstance(sub, PeftModel):
            return sub
    return None


def load_checkpoint(model: MLLMEarlyDisentangleGaussMIModel, checkpoint_path: str, rank: int,
                    load_lora: bool = False, grpo_checkpoint: str = None, grpo_lora_cfg: dict = None):
    from peft import LoraConfig, get_peft_model, PeftModel

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 1. Load trainable modules
    for mname in TRAINABLE_MODULES:
        if mname in checkpoint:
            getattr(model, mname).load_state_dict(checkpoint[mname])
            if rank == 0:
                logger.info(f"✓ {mname} loaded")
        else:
            if rank == 0:
                logger.warning(f"⚠ {mname} not found in checkpoint")

    # 2. Handle LoRA loading (dual adapter stacking)
    llm_model = model.backbone.model
    has_grpo = grpo_checkpoint and os.path.exists(grpo_checkpoint) and grpo_lora_cfg

    if load_lora and 'lora' in checkpoint:
        lora_state = checkpoint['lora']
        loaded = 0

        if rank == 0:
            ckpt_keys = list(lora_state.keys())[:3]
            model_keys = [n for n, _ in model.backbone.named_parameters() if 'lora' in n.lower()][:3]
            logger.info(f"[DEBUG] ckpt lora keys: {ckpt_keys}")
            logger.info(f"[DEBUG] model lora keys: {model_keys}")

        # Three-pass matching to load SFT LoRA into default adapter
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

        if loaded == 0:
            for n, p in model.backbone.named_parameters():
                if 'lora' not in n.lower():
                    continue
                for ck, cv in lora_state.items():
                    if n.endswith(ck) or ck.endswith(n) or ('model.' + ck) == n:
                        p.data.copy_(cv)
                        loaded += 1
                        break

        if rank == 0:
            logger.info(f"✓ SFT LoRA loaded into default adapter ({loaded} params)")

        if has_grpo:
            # SFT+GRPO: rename default → sft + create grpo adapter
            peft_model = _get_peft_model(llm_model)
            if peft_model is None:
                raise RuntimeError("Dual adapter mode requires PeftModel, but none found")

            # Create sft adapter and copy default weights
            sft_lora_config = peft_model.peft_config['default']
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

            peft_model.delete_adapter('default')

            # Create grpo adapter
            grpo_lora_config = LoraConfig(
                r=grpo_lora_cfg['r'],
                lora_alpha=grpo_lora_cfg['lora_alpha'],
                lora_dropout=grpo_lora_cfg.get('lora_dropout', 0.0),
                target_modules=grpo_lora_cfg['target_modules'],
                bias=grpo_lora_cfg.get('bias', 'none'),
                task_type="CAUSAL_LM",
            )
            peft_model.add_adapter(GRPO_ADAPTER, grpo_lora_config)

            # Load GRPO LoRA weights
            grpo_ckpt = torch.load(grpo_checkpoint, map_location='cpu')
            if 'lora' not in grpo_ckpt:
                raise RuntimeError("'lora' weights not found in GRPO checkpoint")
            grpo_lora_state = grpo_ckpt['lora']

            grpo_loaded = 0
            for n, p in model.backbone.model.named_parameters():
                if n in grpo_lora_state:
                    p.data.copy_(grpo_lora_state[n])
                    grpo_loaded += 1
            if grpo_loaded == 0:
                for n, p in model.backbone.model.named_parameters():
                    short_key = n[len('model.'):] if n.startswith('model.') else n
                    if short_key in grpo_lora_state:
                        p.data.copy_(grpo_lora_state[short_key])
                        grpo_loaded += 1

            # Activate dual adapters
            model.backbone._use_lora = True
            model.backbone.set_active_adapters([SFT_ADAPTER, GRPO_ADAPTER])

            if rank == 0:
                logger.info(f"✓ GRPO LoRA loaded ({grpo_loaded} params)")
                logger.info(f"✓ Dual adapters active: [{SFT_ADAPTER}, {GRPO_ADAPTER}]")

            # Override trainable modules from GRPO checkpoint
            for mname in TRAINABLE_MODULES:
                if mname in grpo_ckpt:
                    getattr(model, mname).load_state_dict(grpo_ckpt[mname])
                    if rank == 0:
                        logger.info(f"✓ {mname} overridden from GRPO checkpoint")

    elif has_grpo:
        # GRPO only (no SFT LoRA)
        grpo_lora_config = LoraConfig(
            r=grpo_lora_cfg['r'],
            lora_alpha=grpo_lora_cfg['lora_alpha'],
            lora_dropout=grpo_lora_cfg.get('lora_dropout', 0.0),
            target_modules=grpo_lora_cfg['target_modules'],
            bias=grpo_lora_cfg.get('bias', 'none'),
            task_type="CAUSAL_LM",
        )

        peft_model = llm_model.model
        if isinstance(peft_model, PeftModel):
            if 'default' in peft_model.peft_config:
                peft_model.delete_adapter('default')
            peft_model.add_adapter(GRPO_ADAPTER, grpo_lora_config)
        else:
            llm_model.model = get_peft_model(llm_model.model, grpo_lora_config, adapter_name=GRPO_ADAPTER)
            model.backbone.model = llm_model

        grpo_ckpt = torch.load(grpo_checkpoint, map_location='cpu')
        if 'lora' not in grpo_ckpt:
            raise RuntimeError("'lora' weights not found in GRPO checkpoint")
        grpo_lora_state = grpo_ckpt['lora']

        grpo_loaded = 0
        for n, p in model.backbone.model.named_parameters():
            if n in grpo_lora_state:
                p.data.copy_(grpo_lora_state[n])
                grpo_loaded += 1
        if grpo_loaded == 0:
            for n, p in model.backbone.model.named_parameters():
                short_key = n[len('model.'):] if n.startswith('model.') else n
                if short_key in grpo_lora_state:
                    p.data.copy_(grpo_lora_state[short_key])
                    grpo_loaded += 1

        model.backbone._use_lora = True
        model.backbone.set_active_adapters([GRPO_ADAPTER])

        if rank == 0:
            logger.info(f"✓ GRPO LoRA loaded ({grpo_loaded} params, no SFT LoRA)")

        # Override trainable modules from GRPO checkpoint
        for mname in TRAINABLE_MODULES:
            if mname in grpo_ckpt:
                getattr(model, mname).load_state_dict(grpo_ckpt[mname])
                if rank == 0:
                    logger.info(f"✓ {mname} overridden from GRPO checkpoint")

    if rank == 0:
        epoch = checkpoint.get('epoch', '?')
        val_loss = checkpoint.get('val_loss', '?')
        logger.info(f"Checkpoint: epoch={epoch}, val_loss={val_loss}")


def run_inference(config_path: str, checkpoint_path: str, test_file: str, output_file: str,
                  load_lora: bool = False, grpo_checkpoint: str = None):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        logger.info("=" * 70)
        logger.info("MLLM Early-Style Disentangle Inference (Gaussian · CCA-MI)")
        logger.info(f"Checkpoint: {checkpoint_path}")
        logger.info(f"Test file:  {test_file}")
        logger.info(f"Output:     {output_file}")
        logger.info(f"World size: {world_size}, Rank: {rank}")
        logger.info("=" * 70)

    model_cfg = config['model']
    backbone_type = model_cfg['backbone_type']

    if rank == 0:
        logger.info(f"Loading {backbone_type} backbone...")

    backbone = build_mllm_backbone(
        backbone_type=backbone_type,
        model_path=model_cfg['model_path'],
        use_flash_attention=model_cfg.get('use_flash_attention', False),
        use_gradient_checkpointing=False,
        use_lora=load_lora or model_cfg.get('use_lora', False),
        lora_config=model_cfg.get('lora') if (load_lora or model_cfg.get('use_lora', False)) else None,
        freeze_visual_encoder=True,
    )

    model = MLLMEarlyDisentangleGaussMIModel(
        backbone=backbone,
        disentangle_config=model_cfg.get('disentangle', {}),
    )

    # Prepare GRPO LoRA config (adapter creation and loading happen in load_checkpoint)
    grpo_lora_cfg = None
    if grpo_checkpoint and os.path.exists(grpo_checkpoint):
        grpo_lora_cfg = model_cfg.get('grpo_lora', {
            'r': 8, 'lora_alpha': 16, 'lora_dropout': 0.0,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        })
        if rank == 0:
            logger.info("GRPO checkpoint specified, will be handled in load_checkpoint")

    load_checkpoint(model, checkpoint_path, rank, load_lora=load_lora,
                    grpo_checkpoint=grpo_checkpoint, grpo_lora_cfg=grpo_lora_cfg)

    # Freeze entire model (inference)
    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device)
    # Align new trainable modules to backbone dtype for inference consistency
    target_dtype = next(model.backbone.model.parameters()).dtype
    for mname in TRAINABLE_MODULES:
        module = getattr(model, mname)
        module.to(device=device, dtype=target_dtype)
    model.eval()

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    processor = backbone.get_processor()
    tokenizer = backbone.get_tokenizer()
    data_cfg = config['data']
    inf_cfg = config.get('inference', {})
    no_cot = config.get('no_cot', False)
    DatasetClass = MLLMNoCotDataset if no_cot else MLLMDataset
    _extract = extract_question_no_cot if no_cot else extract_question

    if rank == 0:
        if no_cot:
            logger.info("no_cot=True, using MLLMNoCotDataset + simplified extract logic")
        logger.info(f"Loading test dataset: {test_file}")

    test_dataset = DatasetClass(
        data_file=test_file,
        image_root=data_cfg['image_root'],
        processor=processor,
        tokenizer=tokenizer,
        backbone_type=backbone_type,
        max_constraint_length=data_cfg.get('max_constraint_length', 256),
        max_target_length=data_cfg.get('max_target_length', 512),
        filter_non_multimodal=data_cfg.get('filter_non_multimodal', True),
        preprocess_workers=data_cfg.get('preprocess_workers', 4),
        inference_mode=True,
    )

    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank,
        shuffle=False, drop_last=False,
    ) if world_size > 1 else None

    collate = partial(
        mllm_collate_fn,
        backbone_type=backbone_type,
        pad_token_id=tokenizer.pad_token_id or 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=inf_cfg.get('batch_size', 4),
        sampler=test_sampler,
        shuffle=False,
        num_workers=4,
        collate_fn=collate,
        pin_memory=True,
    )

    if rank == 0:
        logger.info(f"Test set size: {len(test_dataset)}")

    max_new_tokens = inf_cfg.get('max_new_tokens', 256)
    num_beams = inf_cfg.get('num_beams', 1)
    do_sample = inf_cfg.get('do_sample', False)
    max_retries = inf_cfg.get('max_retries', 3)

    if rank == 0:
        logger.info(
            f"Inference params: max_new_tokens={max_new_tokens}, num_beams={num_beams}, "
            f"do_sample={do_sample}, max_retries={max_retries}"
        )

    predictions = []
    current_idx = rank if test_sampler else 0
    stride = world_size if test_sampler else 1

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Inference") if rank == 0 else test_loader
        for batch in pbar:
            ids = batch.pop('id', None)
            answers = batch.pop('answer', [])
            question_types = batch.pop('question_type', [])

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            encode_kwargs = {}
            if 'image_grid_thw' in batch:
                encode_kwargs['image_grid_thw'] = batch.pop('image_grid_thw')
            if 'image_sizes' in batch:
                encode_kwargs['image_sizes'] = batch.pop('image_sizes')
            if 'num_patches_list' in batch:
                encode_kwargs['num_patches_list'] = batch.pop('num_patches_list')

            inner = model.module if hasattr(model, 'module') else model
            for retry in range(max_retries):
                generated_texts = inner.generate(
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
                    num_beams=num_beams,
                    do_sample=do_sample,
                )
                if all(_extract(t) for t in generated_texts):
                    break
                if retry < max_retries - 1:
                    logger.warning(f"Generation {retry + 1} contains empty results, retrying...")
                else:
                    logger.warning(f"Max retries {max_retries} reached, using last generation")

            for i, text in enumerate(generated_texts):
                predictions.append({
                    'dataset_idx': current_idx,
                    'id': ids[i] if ids else str(current_idx),
                    'full_output': text,
                    'predict': _extract(text),
                    'answer': answers[i] if i < len(answers) else '',
                    'question_type': question_types[i] if i < len(question_types) else '',
                })
                current_idx += stride

    logger.info(f"Rank {rank}: collected {len(predictions)} predictions")

    # Multi-GPU aggregation
    if world_size > 1:
        dist.barrier()
        if rank == 0:
            gathered = [None] * world_size
            dist.gather_object(predictions, gathered, dst=0)
            all_preds = [p for rank_preds in gathered for p in rank_preds]
            all_preds.sort(key=lambda x: x['dataset_idx'])

            seen, unique = set(), []
            for p in all_preds:
                if p['id'] not in seen:
                    seen.add(p['id'])
                    unique.append(p)
            if len(unique) < len(all_preds):
                logger.info(f"Dedup removed {len(all_preds) - len(unique)} duplicate samples")
            predictions = unique
        else:
            dist.gather_object(predictions, dst=0)

    # Save results
    if rank == 0:
        if len(predictions) != len(test_dataset):
            logger.warning(f"Prediction count mismatch: expected {len(test_dataset)}, got {len(predictions)}")
        else:
            logger.info(f"Prediction count verified: {len(predictions)}")

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(test_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f) if test_file.endswith('.json') else [json.loads(l) for l in f]
        id_to_label = {str(s.get('id', '')): s.get('question', '') for s in original_data}

        with open(output_path, 'w', encoding='utf-8') as f:
            for p in predictions:
                record = {
                    'id': p['id'],
                    'predict': p['predict'],
                    'label': id_to_label.get(str(p['id']), ''),
                    'full_output': p['full_output'],
                    'answer': p['answer'],
                    'question_type': p['question_type'],
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        logger.info(f"Predictions saved to {output_path}")
        logger.info("Inference complete")


def main():
    parser = argparse.ArgumentParser(description="MLLM Early-Style Disentangle Inference (Gaussian · CCA-MI)")
    parser.add_argument('--config', type=str, required=True, help="Config file path")
    parser.add_argument('--checkpoint', type=str, required=True, help="SFT checkpoint file path")
    parser.add_argument('--test_file', type=str, required=True, help="Test data file path")
    parser.add_argument('--output_file', type=str, required=True, help="Prediction output path")
    parser.add_argument('--local_rank', type=int, default=-1, help="Auto-set by torchrun")
    parser.add_argument('--load_lora', action='store_true', help="Load LoRA weights from SFT checkpoint")
    parser.add_argument('--grpo_checkpoint', type=str, default=None,
                        help="GRPO checkpoint path (stacked LoRA loading)")
    args = parser.parse_args()

    run_inference(args.config, args.checkpoint, args.test_file, args.output_file,
                  load_lora=args.load_lora, grpo_checkpoint=args.grpo_checkpoint)


if __name__ == '__main__':
    main()
