#!/usr/bin/env python
"""
MLLM Early-Style Disentangle Training Script (BA fixed-Gaussian + CLUB all-pairs · DeepSpeed ZeRO-3)

Architecture: MLLM built-in ViT (post-merger) + LLM encode_text(n_text_layers)
              -> shared_dual / visual_private / text_private produce deterministic h
              -> concatenate [h_vp | h_sI | h_sT | h_tp] + segment_emb, feed to LLM
              -> side pathway: pool + L2 normalize -> BA alignment + CLUB all-pairs separation

Usage:
    CUDA_VISIBLE_DEVICES=0 deepspeed scripts/train_mllm_early_disentangle_gauss_mi.py \\
        --config config/mllm_early_disentangle_gauss_mi.yaml \\
        --deepspeed config/deepspeed_early_disentangle.json
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import yaml
import random
import numpy as np
import torch
import deepspeed
import json
import matplotlib
matplotlib.use('Agg')  # Enable plotting without X server
import matplotlib.pyplot as plt
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import logging
from pathlib import Path

from src.models.mllm_backbone import build_mllm_backbone
from src.models.mllm_early_disentangle_gauss_mi_model import MLLMEarlyDisentangleGaussMIModel
from src.data.mllm_dataset import MLLMDataset, mllm_collate_fn
from src.data.mllm_ablation_no_cot_dataset import MLLMNoCotDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# New module names for training / checkpoint saving
TRAINABLE_MODULES = [
    'shared_dual',
    'visual_private',
    'text_private',
    'segment_embedding',
    'visual_recon_decoder',
    'text_recon_decoder',
]

# (yaml freeze config key, related module names) -- for fine-grained freeze control
FREEZE_GROUPS = [
    ('freeze_shared_dual',          ['shared_dual']),
    ('freeze_visual_private',       ['visual_private']),
    ('freeze_text_private',         ['text_private']),
    ('freeze_segment_emb',          ['segment_embedding']),
    ('freeze_recon_decoder',        ['visual_recon_decoder', 'text_recon_decoder']),
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_distributed():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    return rank, world_size, local_rank


def configure_freezing(model: MLLMEarlyDisentangleGaussMIModel, config: dict, rank: int):
    """Configure module freezing strategy"""
    training_cfg = config['training']

    if training_cfg.get('freeze_backbone_llm', True):
        for name, param in model.backbone.named_parameters():
            if 'lora' not in name.lower():
                param.requires_grad = False
        if rank == 0:
            logger.info("  - Backbone (ViT + LLM): frozen (non-LoRA parts)")

    for cfg_key, module_names in FREEZE_GROUPS:
        frozen = training_cfg.get(cfg_key, False)
        for mname in module_names:
            module = getattr(model, mname)
            for p in module.parameters():
                p.requires_grad = (not frozen)
            if rank == 0:
                logger.info(f"  - {mname}: {'frozen' if frozen else 'trainable'}")

    if rank == 0:
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in model.parameters())
        logger.info(
            f"\nTrainable params: {num_trainable:,} / {num_total:,} "
            f"({num_trainable/num_total*100:.1f}%)"
        )


def prepare_optimizer_params(model: MLLMEarlyDisentangleGaussMIModel, config: dict, rank: int):
    """Prepare optimizer parameter groups with layered learning rates"""
    lr_cfg = config['training']['learning_rate']
    training_cfg = config['training']
    groups = []

    # All new modules use new_modules lr; id dedup prevents shared submodules from being added twice
    seen_param_ids = set()
    new_module_params = []
    for mname in TRAINABLE_MODULES:
        module = getattr(model, mname)
        for p in module.parameters():
            if p.requires_grad and id(p) not in seen_param_ids:
                new_module_params.append(p)
                seen_param_ids.add(id(p))
    if new_module_params:
        groups.append({
            "params": new_module_params,
            "lr": lr_cfg.get('new_modules', 2e-4),
            "name": "new_modules",
        })

    if training_cfg.get('train_lora', False):
        lora_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and 'lora' in n.lower()]
        if lora_params:
            groups.append({
                "params": lora_params,
                "lr": lr_cfg.get('lora', 1e-5),
                "name": "lora",
            })

    if rank == 0:
        for g in groups:
            logger.info(f"  Optimizer group [{g['name']}]: {len(g['params'])} params, lr={g['lr']:.2e}")

    return groups


# Separation pair predictor names (matching model vclub_predictors keys)
PREDICTOR_PAIRS = ['vp_si', 'tp_st', 'vp_st', 'tp_si', 'vp_tp']

# SSD trainable modules (backbone handled separately by configure_freezing)
SSD_MODULES = TRAINABLE_MODULES[:]


def set_predictor_trainable(model: MLLMEarlyDisentangleGaussMIModel, trainable: bool):
    """Set requires_grad for 5 vMF predictors (unfrozen in Phase B, frozen in A/C theta step)"""
    for key in PREDICTOR_PAIRS:
        for p in model.vclub_predictors[key].parameters():
            p.requires_grad = trainable


def set_ssd_trainable(model: MLLMEarlyDisentangleGaussMIModel, trainable: bool):
    """Set requires_grad for SSD modules (shared/private/segment/recon); excludes backbone"""
    for mname in SSD_MODULES:
        for p in getattr(model, mname).parameters():
            p.requires_grad = trainable


def load_checkpoint_if_specified(model: MLLMEarlyDisentangleGaussMIModel, config: dict, rank: int):
    ckpt_path = config['training'].get('load_pretrained')
    if not ckpt_path or not os.path.exists(ckpt_path):
        return
    if rank == 0:
        logger.info(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    for mname in TRAINABLE_MODULES:
        if mname in checkpoint:
            getattr(model, mname).load_state_dict(checkpoint[mname])
            if rank == 0:
                logger.info(f"  ✓ {mname}")

    if 'vclub_predictors' in checkpoint:
        model.vclub_predictors.load_state_dict(checkpoint['vclub_predictors'])
        if rank == 0:
            logger.info("  ✓ vclub_predictors")

    if 'lora' in checkpoint:
        lora_state = checkpoint['lora']
        loaded, missing = 0, 0
        for n, p in model.backbone.named_parameters():
            if n in lora_state:
                p.data.copy_(lora_state[n])
                loaded += 1
            elif 'lora' in n.lower():
                missing += 1
        if rank == 0:
            logger.info(f"  ✓ lora ({loaded} params loaded, {missing} missing)")


def check_mi_config_sanity(model: MLLMEarlyDisentangleGaussMIModel, batch_size: int, rank: int):
    """Startup configuration sanity check"""
    if rank != 0:
        return
    if batch_size < 4:
        logger.warning(
            f"⚠ batch_size ({batch_size}) < 4, too few negative samples for CLUB all-pairs, recommend ≥8"
        )
    logger.info(f"[MI config] batch_size={batch_size}, H={model.backbone.hidden_dim}")


# Train/val metric columns (ordered for plotting)
_LOSS_KEYS_TOTAL = ['total', 'gen', 'dis', 'recon']
_LOSS_KEYS_COMP  = ['align', 'sep']

# history key -> outputs dict key mapping
_STEP_KEY_MAP = {
    'total': 'loss', 'gen': 'generation_loss', 'dis': 'disentangle_loss',
    'recon': 'recon_loss', 'align': 'align_loss', 'sep': 'sep_loss',
}


def plot_loss_curves(history: dict, log_dir: Path):
    """
    Plot 4 PNGs:
        Step-level (train only, thin lines):
            loss_curves_step_total.png      : total / gen / dis
            loss_curves_step_components.png : align / sep
        Epoch-level (train avg + val):
            loss_curves_epoch_total.png      : total / gen / dis
            loss_curves_epoch_components.png : align / sep
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    def _plot_step(keys, title, filename):
        n = len(keys)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
        for i, k in enumerate(keys):
            ax = axes[0, i]
            sx = history.get('train_steps', [])
            sy = history.get(f'train_step_{k}', [])
            if sx:
                ax.plot(sx, sy, color='steelblue', linewidth=0.6)
            ax.set_title(k)
            ax.set_xlabel('step')
            ax.set_ylabel('loss')
            ax.grid(True, alpha=0.3)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = log_dir / filename
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return out

    def _plot_epoch(keys, title, filename):
        n = len(keys)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
        for i, k in enumerate(keys):
            ax = axes[0, i]
            ex = history.get('train_epochs', [])
            ey_t = history.get(f'train_{k}', [])
            ey_v = history.get(f'val_{k}', [])
            if ex:
                ax.plot(ex, ey_t, label='train', color='steelblue', linewidth=2.0, marker='o', markersize=4)
                ax.plot(ex, ey_v, label='val',   color='darkorange', linewidth=2.0, marker='s', markersize=5)
            ax.set_title(k)
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = log_dir / filename
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return out

    ps1 = _plot_step(_LOSS_KEYS_TOTAL, 'Step-level: Total / Gen / Dis / Recon', 'loss_curves_step_total.png')
    ps2 = _plot_step(_LOSS_KEYS_COMP,  'Step-level: Components',               'loss_curves_step_components.png')
    pe1 = _plot_epoch(_LOSS_KEYS_TOTAL, 'Epoch-level: Total / Gen / Dis / Recon', 'loss_curves_epoch_total.png')
    pe2 = _plot_epoch(_LOSS_KEYS_COMP,  'Epoch-level: Components',               'loss_curves_epoch_components.png')
    return ps1, ps2, pe1, pe2


def save_loss_history(history: dict, log_dir: Path):
    """Save raw history as JSON for later re-plotting or comparison."""
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / 'loss_history.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return out


def train(config_path: str, deepspeed_config: str):
    rank, world_size, local_rank = setup_distributed()

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    set_seed(config.get('seed', 42))

    if rank == 0:
        logger.info("=" * 80)
        logger.info("MLLM Early-Style Disentangle Training (Gaussian · CCA-MI + Memory Queue)")
        logger.info(f"World size: {world_size}, Rank: {rank}")
        logger.info("=" * 80)

    model_cfg = config['model']
    backbone_type = model_cfg['backbone_type']

    if rank == 0:
        logger.info(f"\nCreating {backbone_type} backbone...")

    backbone = build_mllm_backbone(
        backbone_type=backbone_type,
        model_path=model_cfg['model_path'],
        use_flash_attention=model_cfg.get('use_flash_attention', False),
        use_gradient_checkpointing=model_cfg.get('use_gradient_checkpointing', False),
        use_lora=model_cfg.get('use_lora', False),
        lora_config=model_cfg.get('lora'),
        freeze_visual_encoder=model_cfg.get('freeze_visual_encoder', True),
    )

    model = MLLMEarlyDisentangleGaussMIModel(
        backbone=backbone,
        disentangle_config=model_cfg.get('disentangle', {}),
    )

    load_checkpoint_if_specified(model, config, rank)
    configure_freezing(model, config, rank)
    check_mi_config_sanity(model, config['training']['batch_size'], rank)

    processor = backbone.get_processor()
    tokenizer = backbone.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    data_cfg = config['data']
    no_cot = config.get('no_cot', False)
    DatasetClass = MLLMNoCotDataset if no_cot else MLLMDataset
    if rank == 0 and no_cot:
        logger.info("no_cot=True, using MLLMNoCotDataset (question-only generation, skip reasoning chain)")

    train_dataset = DatasetClass(
        data_file=data_cfg['train_file'],
        image_root=data_cfg['image_root'],
        processor=processor,
        tokenizer=tokenizer,
        backbone_type=backbone_type,
        max_constraint_length=data_cfg.get('max_constraint_length', 256),
        max_target_length=data_cfg.get('max_target_length', 512),
        max_samples=data_cfg.get('max_samples'),
        filter_non_multimodal=data_cfg.get('filter_non_multimodal', True),
        preprocess_workers=data_cfg.get('preprocess_workers', 4),
    )

    val_dataset = DatasetClass(
        data_file=data_cfg['val_file'],
        image_root=data_cfg['image_root'],
        processor=processor,
        tokenizer=tokenizer,
        backbone_type=backbone_type,
        max_constraint_length=data_cfg.get('max_constraint_length', 256),
        max_target_length=data_cfg.get('max_target_length', 512),
        filter_non_multimodal=data_cfg.get('filter_non_multimodal', True),
        preprocess_workers=data_cfg.get('preprocess_workers', 4),
        inference_mode=False,
    )

    # Pre-cache vision tower encoding results (effective when vision tower is frozen)
    if config['training'].get('cache_vision_features', False):
        if rank == 0:
            logger.info("Pre-caching vision tower encoding...")
        cache_device = torch.device(f'cuda:{local_rank}')
        backbone.to(cache_device)
        backbone.model.eval()
        train_dataset.cache_vision_features(backbone)
        val_dataset.cache_vision_features(backbone)
        backbone.cpu()
        torch.cuda.empty_cache()
        if rank == 0:
            logger.info("Vision cache complete, will skip vision tower forward during training")

    collate = partial(
        mllm_collate_fn,
        backbone_type=backbone_type,
        pad_token_id=tokenizer.pad_token_id or 0,
    )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    ) if world_size > 1 else None

    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    ) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        collate_fn=collate,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=4,
        collate_fn=collate,
        pin_memory=True,
    )

    if rank == 0:
        logger.info(f"Training set: {len(train_dataset)} samples, Validation set: {len(val_dataset)} samples")

    optimizer_grouped_parameters = prepare_optimizer_params(model, config, rank)

    with open(deepspeed_config, 'r') as f:
        ds_config = json.load(f)

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=optimizer_grouped_parameters,
        config=ds_config,
    )

    # deepspeed bf16 casts all model params to bf16, but disentangle side pathway
    # (_pool_and_normalize uses explicit .float()) is designed for fp32; predictors must
    # stay fp32 consistent with norm_*, otherwise F.layer_norm gets Float/BFloat16 mismatch.
    model.vclub_predictors.float()

    num_epochs = config['training']['num_epochs']
    gen_loss_weight = config['training'].get('gen_loss_weight', 1.0)
    dis_loss_weight = config['training'].get('disentangle_loss_weight', 0.1)
    recon_loss_weight = config['training'].get('recon_loss_weight', 0.1)
    lambda_align = config['training'].get('lambda_align', 1.0)
    kappa_align  = config['training'].get('kappa_align', 1.0)

    # ===== 3-phase vMF-condCLUB schedule (predictor has its own optimizer, independent of deepspeed engine) =====
    phase_a = config['training'].get('phase_a_steps', 500)      # Phase A: shared warm-up, lambda_sep=0
    phase_b = config['training'].get('phase_b_steps', 1000)     # Phase B: predictor warm-up, SSD frozen
    # Stage 2: load stage-1 disentangle + LoRA weights, skip A/B warm-up, go directly to Phase C alternating
    # (predictor not retrained from scratch, uses checkpoint weights as initialization, continues q-step <-> SSD/LoRA-step)
    if config['training'].get('stage2', False):
        phase_a = phase_b = 0
    inner_loop = config['training'].get('phase_c_inner_loop', 3)
    q_lr = config['training'].get('predictor_lr', 1.0e-4)
    config_sep = config['training'].get('lambda_sep', 1.0)      # Phase C separation weight
    q_optimizer = torch.optim.AdamW(
        model.vclub_predictors.parameters(), lr=q_lr,
    )

    save_dir = Path(config['output']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config['output'].get('log_dir', save_dir / 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)

    # Loss curve history (maintained and plotted only on rank 0)
    history = {'train_steps': [], 'train_epochs': []}
    for k in _LOSS_KEYS_TOTAL + _LOSS_KEYS_COMP:
        history[f'train_step_{k}'] = []
        history[f'train_{k}']      = []
        history[f'val_{k}']        = []

    global_step = 0
    best_val_loss = float('inf')

    if rank == 0:
        if config['training'].get('stage2', False):
            logger.info("\n[Stage 2] Loading stage-1 disentangle + LoRA weights, skipping Phase A/B, directly entering Phase C alternating training (SSD + LoRA + predictor continued)")
        else:
            logger.info(f"\n[Stage 1] 3-phase training - {num_epochs} epochs")

    for epoch in range(num_epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        model_engine.train()
        epoch_losses = {
            'total': 0.0, 'gen': 0.0, 'dis': 0.0, 'recon': 0.0,
            'align': 0.0, 'sep': 0.0,
        }
        # Predictor (q) diagnostic accumulation: separate from epoch_losses
        epoch_q = {'sum': 0.0, 'n': 0}
        epoch_q_diag = {k: 0.0 for k in ('vp_si', 'tp_st', 'vp_st', 'tp_si', 'vp_tp', 'cos_mean', 'mu_norm')}
        last_ql = None

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") if rank == 0 else train_loader

        for batch in pbar:
            device = model_engine.device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            batch.pop('id', None)
            batch.pop('answer', None)
            batch.pop('question_type', None)

            encode_kwargs = {}
            if 'image_grid_thw' in batch:
                encode_kwargs['image_grid_thw'] = batch.pop('image_grid_thw')
            if 'image_sizes' in batch:
                encode_kwargs['image_sizes'] = batch.pop('image_sizes')
            if 'num_patches_list' in batch:
                encode_kwargs['num_patches_list'] = batch.pop('num_patches_list')
            if 'cached_v_raw' in batch:
                encode_kwargs['cached_v_raw'] = batch.pop('cached_v_raw')
                encode_kwargs['cached_v_mask'] = batch.pop('cached_v_mask')

            phase = 'A' if global_step < phase_a else (
                'B' if global_step < phase_a + phase_b else 'C'
            )

            if phase == 'A':
                # Shared warm-up: predictor frozen, lambda_sep=0, train alignment only
                set_predictor_trainable(model, False)
                outputs = model_engine.module.train_forward(
                    pixel_values=batch.get('pixel_values'),
                    num_images=batch.get('num_images'),
                    text_input_ids=batch['text_input_ids'],
                    text_attention_mask=batch['text_attention_mask'],
                    system_input_ids=batch['system_input_ids'],
                    system_attention_mask=batch['system_attention_mask'],
                    constraint_input_ids=batch['constraint_input_ids'],
                    constraint_attention_mask=batch['constraint_attention_mask'],
                    target_input_ids=batch['target_input_ids'],
                    target_attention_mask=batch['target_attention_mask'],
                    encode_kwargs=encode_kwargs,
                    gen_loss_weight=gen_loss_weight,
                    disentangle_loss_weight=dis_loss_weight,
                    recon_loss_weight=recon_loss_weight,
                    lambda_align=lambda_align,
                    lambda_sep=0.0,
                    kappa_align=kappa_align,
                )
                loss = outputs['loss']
                model_engine.backward(loss)
                model_engine.step()
                global_step += 1

            elif phase == 'B':
                # Predictor warm-up: SSD frozen, lightweight forward (no LLM generation), train 5 predictors only
                if global_step == phase_a:
                    set_ssd_trainable(model, False)
                    set_predictor_trainable(model, True)
                    if rank == 0:
                        logger.info("==> Phase B: predictor warm-up (SSD frozen, training 5 vMF predictors only)")
                _, _, params = model_engine.module._encode_and_disentangle(
                    batch.get('pixel_values'), batch.get('num_images'),
                    batch['text_input_ids'], batch['text_attention_mask'],
                    encode_kwargs,
                )
                q_optimizer.zero_grad()
                ql = model_engine.module.compute_predictor_loss(params)
                ql.backward()
                q_optimizer.step()
                global_step += 1
                outputs = None
                if rank == 0:
                    last_ql = ql.item()
                    epoch_q['sum'] += last_ql
                    epoch_q['n'] += 1
                    diag = model_engine.module.compute_predictor_diagnostics(params)
                    for k, v in diag['per_pair'].items():
                        epoch_q_diag[k] += v
                    epoch_q_diag['cos_mean'] += diag['cos_mean']
                    epoch_q_diag['mu_norm']  += diag['mu_norm']

                if global_step == phase_a + phase_b:
                    # Exit B: restore SSD trainable, freeze predictors, replay backbone freezing
                    set_predictor_trainable(model, False)
                    set_ssd_trainable(model, True)
                    configure_freezing(model, config, rank)
                    if rank == 0:
                        logger.info("==> Phase C: main training started (predictor inner-loop + SSD joint)")

            else:
                # Phase C: phi (predictor inner-loop) first, then theta (SSD joint). phi must precede theta:
                # theta's L_sep depends on predictor weights; if theta graph is built before updating predictor
                # in-place, backward sees stale weight versions ("modified by an inplace operation").
                # phi-first lets theta build graph with updated weights; L_sep -> mu_phi(x) -> x -> SSD gradients still valid.
                # inner_loop=0 skips phi step, only does theta (SSD/LoRA) update.
                if inner_loop > 0:
                    set_predictor_trainable(model, True)
                    _, _, params = model_engine.module._encode_and_disentangle(
                        batch.get('pixel_values'), batch.get('num_images'),
                        batch['text_input_ids'], batch['text_attention_mask'],
                        encode_kwargs,
                    )
                    for _ in range(inner_loop):
                        q_optimizer.zero_grad()
                        ql = model_engine.module.compute_predictor_loss(params)
                        ql.backward()
                        q_optimizer.step()

                    if rank == 0:
                        last_ql = ql.item()
                        epoch_q['sum'] += last_ql
                        epoch_q['n'] += 1
                        diag = model_engine.module.compute_predictor_diagnostics(params)
                        for k, v in diag['per_pair'].items():
                            epoch_q_diag[k] += v
                        epoch_q_diag['cos_mean'] += diag['cos_mean']
                        epoch_q_diag['mu_norm']  += diag['mu_norm']

                    set_predictor_trainable(model, False)
                outputs = model_engine.module.train_forward(
                    pixel_values=batch.get('pixel_values'),
                    num_images=batch.get('num_images'),
                    text_input_ids=batch['text_input_ids'],
                    text_attention_mask=batch['text_attention_mask'],
                    system_input_ids=batch['system_input_ids'],
                    system_attention_mask=batch['system_attention_mask'],
                    constraint_input_ids=batch['constraint_input_ids'],
                    constraint_attention_mask=batch['constraint_attention_mask'],
                    target_input_ids=batch['target_input_ids'],
                    target_attention_mask=batch['target_attention_mask'],
                    encode_kwargs=encode_kwargs,
                    gen_loss_weight=gen_loss_weight,
                    disentangle_loss_weight=dis_loss_weight,
                    recon_loss_weight=recon_loss_weight,
                    lambda_align=lambda_align,
                    lambda_sep=config_sep,
                    kappa_align=kappa_align,
                )
                loss = outputs['loss']
                model_engine.backward(loss)
                model_engine.step()
                global_step += 1

            if outputs is not None:
                # Step-level train loss (rank 0 only, call .item() immediately to avoid holding GPU tensor)
                # train_steps appended here: Phase B uses lightweight predictor warm-up with outputs=None,
                # only advancing global_step without recording loss; appending unconditionally outside
                # would make x-axis longer than y-axis, causing length mismatch when plotting.
                if rank == 0:
                    history['train_steps'].append(global_step)
                    for k in _LOSS_KEYS_TOTAL + _LOSS_KEYS_COMP:
                        v = outputs[_STEP_KEY_MAP[k]]
                        history[f'train_step_{k}'].append(v.item() if torch.is_tensor(v) else v)

                epoch_losses['total']  += loss.item()
                epoch_losses['gen']    += outputs['generation_loss']
                epoch_losses['dis']    += outputs['disentangle_loss']
                epoch_losses['recon']  += outputs['recon_loss']
                epoch_losses['align']  += outputs['align_loss']
                epoch_losses['sep']    += outputs['sep_loss']

                if rank == 0 and isinstance(pbar, tqdm):
                    postfix = {
                        'loss':  f'{loss.item():.4f}',
                        'gen':   f'{outputs["generation_loss"]:.4f}',
                        'dis':   f'{outputs["disentangle_loss"]:.4f}',
                        'align': f'{outputs["align_loss"]:.3f}',
                        'sep':   f'{outputs["sep_loss"]:.3f}',
                    }
                    if last_ql is not None:   # only show q in Phase B/C
                        postfix['q'] = f'{last_ql:.4f}'
                    pbar.set_postfix(postfix)

        n = len(train_loader)
        for k in epoch_losses:
            epoch_losses[k] /= n

        torch.cuda.empty_cache()

        if rank == 0:
            logger.info(
                f"\nEpoch {epoch+1} training: "
                f"Total={epoch_losses['total']:.4f}, "
                f"Gen={epoch_losses['gen']:.4f}, "
                f"Dis={epoch_losses['dis']:.4f}, "
                f"Recon={epoch_losses['recon']:.4f}, "
                f"Align={epoch_losses['align']:.4f}, "
                f"Sep={epoch_losses['sep']:.4f}"
            )
            if epoch_q['n'] > 0:
                q_mean = epoch_q['sum'] / epoch_q['n']
                d = {k: v / epoch_q['n'] for k, v in epoch_q_diag.items()}
                logger.info(
                    f"Predictor diagnostics (n={epoch_q['n']}): Q={q_mean:.4f} | "
                    f"vp_si={d['vp_si']:.4f} tp_st={d['tp_st']:.4f} "
                    f"vp_st={d['vp_st']:.4f} tp_si={d['tp_si']:.4f} vp_tp={d['vp_tp']:.4f} | "
                    f"mu·y cos={d['cos_mean']:.4f} | mu norm={d['mu_norm']:.4f}"
                )

        # Validation
        if (epoch + 1) % config['output'].get('save_every_n_epochs', 1) == 0:
            model_engine.eval()
            val_losses = {k: 0.0 for k in epoch_losses}

            with torch.no_grad():
                for batch in val_loader:
                    device = model_engine.device
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    batch.pop('id', None)
                    batch.pop('answer', None)
                    batch.pop('question_type', None)

                    encode_kwargs = {}
                    if 'image_grid_thw' in batch:
                        encode_kwargs['image_grid_thw'] = batch.pop('image_grid_thw')
                    if 'image_sizes' in batch:
                        encode_kwargs['image_sizes'] = batch.pop('image_sizes')
                    if 'num_patches_list' in batch:
                        encode_kwargs['num_patches_list'] = batch.pop('num_patches_list')
                    if 'cached_v_raw' in batch:
                        encode_kwargs['cached_v_raw'] = batch.pop('cached_v_raw')
                        encode_kwargs['cached_v_mask'] = batch.pop('cached_v_mask')

                    outputs = model_engine.module.train_forward(
                        pixel_values=batch.get('pixel_values'),
                        num_images=batch.get('num_images'),
                        text_input_ids=batch['text_input_ids'],
                        text_attention_mask=batch['text_attention_mask'],
                        system_input_ids=batch['system_input_ids'],
                        system_attention_mask=batch['system_attention_mask'],
                        constraint_input_ids=batch['constraint_input_ids'],
                        constraint_attention_mask=batch['constraint_attention_mask'],
                        target_input_ids=batch['target_input_ids'],
                        target_attention_mask=batch['target_attention_mask'],
                        encode_kwargs=encode_kwargs,
                        gen_loss_weight=gen_loss_weight,
                        disentangle_loss_weight=dis_loss_weight,
                        recon_loss_weight=recon_loss_weight,
                        lambda_align=lambda_align,
                        lambda_sep=config_sep,
                        kappa_align=kappa_align,
                    )

                    val_losses['total']  += outputs['loss'].item()
                    val_losses['gen']    += outputs['generation_loss']
                    val_losses['dis']    += outputs['disentangle_loss']
                    val_losses['recon']  += outputs['recon_loss']
                    val_losses['align']  += outputs['align_loss']
                    val_losses['sep']    += outputs['sep_loss']

            n_val = len(val_loader)
            for k in val_losses:
                val_losses[k] /= n_val

            device = model_engine.device
            if world_size > 1:
                keys = list(val_losses.keys())
                val_loss_t = torch.tensor(
                    [val_losses[k] for k in keys], dtype=torch.float32, device=device,
                )
                torch.distributed.all_reduce(val_loss_t, op=torch.distributed.ReduceOp.SUM)
                val_loss_t /= world_size
                for i, k in enumerate(keys):
                    val_losses[k] = val_loss_t[i].item()

            if rank == 0:
                logger.info(
                    f"Epoch {epoch+1} validation: "
                    f"Total={val_losses['total']:.4f}, "
                    f"Gen={val_losses['gen']:.4f}, "
                    f"Dis={val_losses['dis']:.4f}"
                )

                # Record epoch-level train/val loss and update curve plots
                history['train_epochs'].append(epoch + 1)
                for k in _LOSS_KEYS_TOTAL + _LOSS_KEYS_COMP:
                    history[f'train_{k}'].append(epoch_losses[k])
                    history[f'val_{k}'].append(val_losses[k])
                ps1, ps2, pe1, pe2 = plot_loss_curves(history, log_dir)
                save_loss_history(history, log_dir)
                logger.info(f"Loss curves updated: {ps1.name}/{ps2.name}/{pe1.name}/{pe2.name} -> {log_dir}")

            is_best = val_losses['total'] < best_val_loss
            if world_size > 1:
                is_best_t = torch.tensor([int(is_best)], device=device)
                torch.distributed.broadcast(is_best_t, src=0)
                is_best = bool(is_best_t.item())

            if is_best:
                best_val_loss = val_losses['total']
                if rank == 0:
                    logger.info("New best model, saving checkpoint...")

                save_path = save_dir / "trainable_best.pt"
                checkpoint_dict = {'epoch': epoch + 1, 'val_loss': best_val_loss}

                for mname in TRAINABLE_MODULES:
                    module = getattr(model_engine.module, mname)
                    with deepspeed.zero.GatheredParameters(
                        list(module.parameters()), modifier_rank=0
                    ):
                        if rank == 0:
                            checkpoint_dict[mname] = module.state_dict()

                # vMF predictors (regular params, no ZeRO shard, but use Gathered mode for consistency)
                with deepspeed.zero.GatheredParameters(
                    list(model_engine.module.vclub_predictors.parameters()), modifier_rank=0
                ):
                    if rank == 0:
                        checkpoint_dict['vclub_predictors'] =\
                            model_engine.module.vclub_predictors.state_dict()

                if rank == 0:
                    checkpoint_dict['q_optimizer_state'] = q_optimizer.state_dict()

                if config['training'].get('train_lora', False):
                    lora_params = {
                        n: p for n, p in model_engine.module.backbone.named_parameters()
                        if 'lora' in n.lower()
                    }
                    with deepspeed.zero.GatheredParameters(
                        list(lora_params.values()), modifier_rank=0
                    ):
                        if rank == 0:
                            checkpoint_dict['lora'] = {
                                n: p.data.clone() for n, p in lora_params.items()
                            }

                if rank == 0:
                    torch.save(checkpoint_dict, save_path)
                    logger.info(f"Checkpoint saved to {save_path}")

    if rank == 0:
        logger.info("Training complete")


def main():
    parser = argparse.ArgumentParser(description="MLLM Early-Style Disentangle Training (Gaussian · CCA-MI)")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--deepspeed', type=str, required=True)
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()
    train(args.config, args.deepspeed)


if __name__ == '__main__':
    main()
