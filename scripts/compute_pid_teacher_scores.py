#!/usr/bin/env python
"""
Gaussian-PID Teacher Score Offline Computation Script

Pipeline:
    1. Load frozen disentangle model
    2. Load/train PIDProjector (project to low-dimensional space)
    3. First pass: iterate dataset to accumulate covariance statistics
    4. Second pass: compute per-sample teacher score
    5. Save {sample_id: teacher_score}

Usage:
    python scripts/compute_pid_teacher_scores.py \
        --config config/pid_reward_model.yaml
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import yaml
import pickle
import torch
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from src.models.mllm_backbone import build_mllm_backbone
from src.models.mllm_early_disentangle_model import MLLMEarlyDisentangleModel
from src.data.mllm_dataset import MLLMDataset, mllm_collate_fn
from src.reward.pid_reward_model import PCAProjector, GaussianPIDScorer, vMFPIDScorer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TRAINABLE_MODULES = [
    'shared_dual', 'visual_private', 'text_private', 'segment_embedding',
]


def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(x.dtype)
    return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)


def load_disentangle_model(config: dict, device: torch.device):
    """Load frozen disentangle model"""
    model_cfg = config['model']
    backbone_type = model_cfg['backbone_type']

    backbone = build_mllm_backbone(
        backbone_type=backbone_type,
        model_path=model_cfg['model_path'],
        use_flash_attention=model_cfg.get('use_flash_attention', False),
        use_gradient_checkpointing=False,
        use_lora=False,
        freeze_visual_encoder=True,
    )

    dis_cfg = model_cfg.get('disentangle', {})
    model = MLLMEarlyDisentangleModel(backbone=backbone, disentangle_config=dis_cfg)

    # Load checkpoint
    ckpt_path = config['pid_reward_model']['disentangle_checkpoint']
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    for mname in TRAINABLE_MODULES:
        if mname in checkpoint:
            getattr(model, mname).load_state_dict(checkpoint[mname])
            logger.info(f"✓ {mname} loaded")

    model = model.float().to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, backbone_type


def extract_features(model, batch, device):
    """Extract PID features (S, V, T, Y) from batch"""
    batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

    encode_kwargs = {}
    if 'image_grid_thw' in batch_gpu:
        encode_kwargs['image_grid_thw'] = batch_gpu.pop('image_grid_thw')

    with torch.no_grad():
        _, _, dis_out = model._encode_and_disentangle(
            pixel_values=batch_gpu['pixel_values'],
            num_images=batch_gpu.get('num_images'),
            text_input_ids=batch_gpu['text_input_ids'],
            text_attention_mask=batch_gpu['text_attention_mask'],
            encode_kwargs=encode_kwargs,
        )

        # Question encoding
        q_hidden = model.backbone.encode_text(
            batch_gpu['target_input_ids'],
            batch_gpu['target_attention_mask'],
            model.n_text_layers,
        )

    # Pool
    z_si = dis_out['v_shared'].mean(dim=1)   # [B, H]
    z_st = dis_out['t_shared'].mean(dim=1)   # [B, H]
    z_vp = masked_mean_pool(dis_out['v_priv'], dis_out['v_mask'])  # [B, H]
    z_tp = masked_mean_pool(dis_out['t_priv'], dis_out['t_mask'])  # [B, H]
    z_y = masked_mean_pool(q_hidden, batch_gpu['target_attention_mask'])  # [B, H]

    S = torch.cat([z_si, z_st], dim=-1)  # [B, 2H]
    return S, z_vp, z_tp, z_y


def main():
    parser = argparse.ArgumentParser(description="Compute PID Teacher Scores")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--sft_config', type=str, default='config/mllm_early_disentangle.yaml',
                        help="SFT stage config file (provides model and data config)")
    parser.add_argument('--mode', type=str, default='vmf', choices=['gaussian', 'vmf'],
                        help="scorer type: gaussian (Gaussian MI) or vmf (vMF conditional likelihood)")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        pid_config = yaml.safe_load(f)
    with open(args.sft_config, 'r', encoding='utf-8') as f:
        sft_config = yaml.safe_load(f)

    config = sft_config.copy()
    config['pid_reward_model'] = pid_config['pid_reward_model']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pid_cfg = pid_config['pid_reward_model']
    reg_scale = pid_cfg.get('reg_scale', 1e-4)

    logger.info("=" * 70)
    if args.mode == 'vmf':
        logger.info("vMF-PID Teacher Score computation (Ridge regression + normalize)")
        logger.info(f"reg_scale={reg_scale}")
    else:
        proj_dim = pid_cfg.get('proj_dim', 256)
        logger.info("Gaussian-PID Teacher Score computation (PCA + Whitening)")
        logger.info(f"proj_dim={proj_dim}, reg_scale={reg_scale}")
    logger.info("=" * 70)

    # 1. Load model
    model, backbone_type = load_disentangle_model(config, device)
    # 2. Build dataset
    processor = model.backbone.get_processor()
    tokenizer = model.backbone.get_tokenizer()
    data_cfg = config['data']

    dataset = MLLMDataset(
        data_file=data_cfg['train_file'],
        image_root=data_cfg['image_root'],
        processor=processor,
        tokenizer=tokenizer,
        backbone_type=backbone_type,
        max_constraint_length=data_cfg.get('max_constraint_length', 256),
        max_target_length=data_cfg.get('max_target_length', 512),
        filter_non_multimodal=data_cfg.get('filter_non_multimodal', True),
        preprocess_workers=data_cfg.get('preprocess_workers', 4),
    )

    collate = partial(
        mllm_collate_fn,
        backbone_type=backbone_type,
        pad_token_id=tokenizer.pad_token_id or 0,
    )

    loader = DataLoader(
        dataset, batch_size=pid_cfg.get('training', {}).get('batch_size', 64),
        shuffle=False, num_workers=4, collate_fn=collate, pin_memory=True,
    )

    logger.info(f"Dataset size: {len(dataset)}")

    # 3. GPU feature extraction
    logger.info("Extracting raw features on GPU...")
    all_feats = []
    all_ids = []
    sample_idx = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting raw features"):
            S, V, T, Y = extract_features(model, batch, device)
            all_feats.append((S.cpu(), V.cpu(), T.cpu(), Y.cpu()))

            ids = batch.get('id', None)
            B = S.size(0)
            for i in range(B):
                all_ids.append(ids[i] if ids is not None else str(sample_idx))
                sample_idx += 1

    S_all = torch.cat([f[0] for f in all_feats])
    V_all = torch.cat([f[1] for f in all_feats])
    T_all = torch.cat([f[2] for f in all_feats])
    Y_all = torch.cat([f[3] for f in all_feats])
    del all_feats
    logger.info(f"Feature extraction complete: S{list(S_all.shape)}, V{list(V_all.shape)}, T{list(T_all.shape)}, Y{list(Y_all.shape)}")

    # 4. Compute teacher scores
    if args.mode == 'vmf':
        scorer = vMFPIDScorer(reg_scale=reg_scale)
        batch_size = pid_cfg.get('training', {}).get('batch_size', 64)
        N = S_all.size(0)
        for i in tqdm(range(0, N, batch_size), desc="Caching samples"):
            end = min(i + batch_size, N)
            scorer.update(S_all[i:end], V_all[i:end], T_all[i:end], Y_all[i:end])
        scorer.finalize()
        scores = scorer.score(S_all, V_all, T_all, Y_all)
    else:
        proj_dim = pid_cfg.get('proj_dim', 256)
        scorer = GaussianPIDScorer(proj_dim=proj_dim, reg_scale=reg_scale)
        logger.info("Fit PCA + whitening...")
        projector = PCAProjector(proj_dim=proj_dim)
        projector.fit(S_all, V_all, T_all, Y_all)
        S_p, V_p, T_p, Y_p = projector(S_all, V_all, T_all, Y_all)
        batch_size = pid_cfg.get('training', {}).get('batch_size', 64)
        N = S_p.size(0)
        for i in tqdm(range(0, N, batch_size), desc="Accumulating statistics"):
            end = min(i + batch_size, N)
            scorer.update(S_p[i:end], V_p[i:end], T_p[i:end], Y_p[i:end])
        scorer.finalize()
        scores = scorer.score(S_p, V_p, T_p, Y_p)

    teacher_scores = {sid: scores[i].item() for i, sid in enumerate(all_ids)}

    # 5. Statistics and save
    import numpy as np
    values = list(teacher_scores.values())
    logger.info(
        f"Teacher score distribution: mean={np.mean(values):.4f}, "
        f"std={np.std(values):.4f}, "
        f"min={np.min(values):.4f}, max={np.max(values):.4f}"
    )

    output_file = pid_cfg.get('teacher_scores_file', 'data/pid_teacher_scores.pkl')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(teacher_scores, f)

    save_dir = pid_cfg.get('save_dir', 'outputs/checkpoints/pid_reward_model')
    os.makedirs(save_dir, exist_ok=True)

    if args.mode == 'vmf':
        scorer_path = os.path.join(save_dir, 'vmf_scorer.pt')
        torch.save({
            'reg_scale': scorer.reg_scale,
            'count': scorer.count,
            '_regressors': scorer._regressors,
        }, scorer_path)
    else:
        projector_path = os.path.join(save_dir, 'pca_projector.pt')
        projector.save(projector_path)
        scorer_path = os.path.join(save_dir, 'scorer.pt')
        torch.save({
            'proj_dim': scorer.proj_dim,
            'reg_scale': scorer.reg_scale,
            'count': scorer.count,
            'mean': scorer.mean,
            'cov': scorer.cov,
            'cross_cov': scorer.cross_cov,
            '_pmi_cache': scorer._pmi_cache,
            '_global_synergy': scorer._global_synergy,
        }, scorer_path)
        logger.info(f"PCA Projector saved to: {projector_path}")

    logger.info(f"Teacher scores saved to: {output_file}")
    logger.info(f"Scorer state saved to: {scorer_path}")
    logger.info("Done")


if __name__ == '__main__':
    main()
