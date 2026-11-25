"""
MLLM Early-Style Disentangle (deterministic H backbone + vMF geodesic alignment + explicit vMF-condCLUB separation)

Theory:
    L2-normalized representations lie on the unit hypersphere -> von Mises-Fisher distribution
    - Alignment: minimize hypersphere geodesic distance arccos^2(cos(s_I, s_T))
    - Separation: explicitly trained vMF conditional predictor q_phi(Y|X), vCLUB = E_joint[log q] - E_product[log q]
      5 pairs: v_priv vs s_I, t_priv vs s_T, v_priv vs s_T, t_priv vs s_I, v_priv vs t_priv

Implementation:
    - Backbone: disentangler -> deterministic H-dim h -> cat and feed to LLM for generation
    - Side pathway: h -> masked mean pool -> L2 normalize -> [B, H] unit vectors
    - Alignment: arccos^2(cos_sim).mean()
    - Separation: 5 independent VMFConditionalPredictors, condCLUB (theta step freezes phi,
      phi step matched-positive trains predictor only)

Loss:
    L = L_gen + lambda_align * L_align + lambda_sep * L_sep
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
import logging

from .mllm_backbone import MLLMBackbone
from .early_disentangle_projectors import (
    _SharedDualResampler,
    GatedPrivateBlock,
    ReconstructionDecoder,
)

logger = logging.getLogger(__name__)

SEG_V_PRIV = 0
SEG_S_I    = 1
SEG_S_T    = 2
SEG_T_PRIV = 3


class VMFConditionalPredictor(nn.Module):
    """Explicitly trained vMF conditional predictor q_phi(Y|X) = vMF(mu_phi(X), kappa).

    mu_phi(X) = normalize(f_phi(X)) -> S^{H-1} unit direction; kappa cancels analytically
    in the joint/product difference of CLUB, no Bessel normalizer needed. Trained on
    matched-positives only (see compute_predictor_loss), generalizes to off-diagonal via MLP.
    """

    def __init__(self, H: int, bottleneck: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, H),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class MLLMEarlyDisentangleGaussMIModel(nn.Module):
    """Deterministic H backbone + vMF alignment + explicit vMF-condCLUB separation"""

    def __init__(
        self,
        backbone: MLLMBackbone,
        disentangle_config: Optional[dict] = None,
    ):
        super().__init__()

        cfg = disentangle_config or {}
        H = backbone.hidden_dim
        n_q   = cfg.get('n_queries_shared', 32)
        num_heads = cfg.get('num_heads', 8)
        dropout   = cfg.get('dropout', 0.1)
        sr_dropout = cfg.get('shared_dual_dropout', 0.1)
        self.n_text_layers = cfg.get('n_text_layers', 4)
        self.kappa_sep = cfg.get('kappa_sep', 1.0)

        self.backbone = backbone
        self.n_queries = n_q
        self.max_text_tokens = cfg.get('max_text_tokens', 512)

        # ===== Backbone =====
        self.shared_dual = _SharedDualResampler(H, n_queries=n_q, num_heads=num_heads, dropout=sr_dropout)
        self.visual_private = GatedPrivateBlock(H, num_heads=num_heads, dropout=dropout)
        self.text_private   = GatedPrivateBlock(H, num_heads=num_heads, dropout=dropout)
        self.segment_embedding = nn.Embedding(4, H)
        nn.init.normal_(self.segment_embedding.weight, mean=0.0, std=0.02)

        # ===== Reconstruction Decoder =====
        self.visual_recon_decoder = ReconstructionDecoder(H, num_heads=num_heads, dropout=dropout)
        self.text_recon_decoder   = ReconstructionDecoder(H, num_heads=num_heads, dropout=dropout)

        # ===== Explicit vMF conditional predictors (one per separation pair, unshared) =====
        # Direction X -> Y:
        #   vp_si: norm_vp -> norm_sI    tp_st: norm_tp -> norm_sT
        #   vp_st: norm_vp -> norm_sT    tp_si: norm_tp -> norm_sI
        #   vp_tp: norm_vp -> norm_tp
        self.vclub_predictors = nn.ModuleDict({
            'vp_si': VMFConditionalPredictor(H),
            'tp_st': VMFConditionalPredictor(H),
            'vp_st': VMFConditionalPredictor(H),
            'tp_si': VMFConditionalPredictor(H),
            'vp_tp': VMFConditionalPredictor(H),
        })

        logger.info(
            "MLLMEarlyDisentangleGaussMIModel(BA + CLUB all-pairs) initialized: "
            f"H={H}, n_queries={n_q}"
        )

    # ==================== Loss Computation ====================

    @staticmethod
    def _pool_and_normalize(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """masked mean pooling + L2 normalize -> [B, H] unit vectors"""
        h = h.float()
        mask_f = mask.to(h.dtype).unsqueeze(-1)
        h_pool = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return F.normalize(h_pool, dim=-1)

    @staticmethod
    def _vmf_align_loss(s_I: torch.Tensor, s_T: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
        """vMF alignment likelihood L_vMF-A (InfoNCE, positive pairs are per-sample i<->i).

        s_I, s_T: (B, H) unit directions. Positive uses self-matching (s_I[i] <-> s_T[i]),
        negative uses the remaining B-1 in-batch samples. kappa is fixed (default 1).

        p(j|s_T,i,S_I) = exp(kappa * s_I,j^T * s_T,i) / sum_k exp(kappa * s_I,k^T * s_T,i)
        L_I->T = -(1/B) sum_i log p(i|s_T,i,S_I)
        L_T->I = -(1/B) sum_j log p(j|s_I,j,S_T)
        L = (L_I->T + L_T->I) / 2
        """
        B = s_I.size(0)
        if B < 2:
            return s_I.new_tensor(0.0)
        logits = kappa * (s_I @ s_T.T)  # (B,B), diagonal = kappa*cos(s_I[i], s_T[i])
        # L_I->T = -(1/B) sum_i log p(i|s_T,i) = mean_i[lse_col[i] - logit[i,i]]
        lse_col = torch.logsumexp(logits, dim=1)
        l_it = (lse_col - logits.diag()).mean()
        # L_T->I = -(1/B) sum_j log p(j|s_I,j) = mean_j[lse_row[j] - logit[j,j]]
        lse_row = torch.logsumexp(logits, dim=0)
        l_ti = (lse_row - logits.diag()).mean()
        return 0.5 * (l_it + l_ti)

    @staticmethod
    def _vclub_pair_loss(priv: torch.Tensor, shared: torch.Tensor, predictor: nn.Module, kappa: float) -> torch.Tensor:
        """Explicitly trained vCLUB: E_joint[log q] - E_product[log q].

        log q(y|x) = kappa * mu_phi(x)^T y - log Z(kappa), Z(kappa) cancels analytically.
        Predictor params must be frozen (requires_grad=False) before calling, but its forward
        retains the computation graph through priv, so L_sep -> mu_phi(x) -> x -> SSD gradients hold.
        shared(Y) serves as the regression target and is detached: separation only pushes the private
        side to disentangle; shared is handled by alignment/generation loss, avoiding gradient chasing.
        Can be negative; no ReLU / clamp / abs applied.
        """
        B = priv.size(0)
        if B < 2:
            return priv.new_tensor(0.0)

        mu = predictor(priv)                
        logq = kappa * (mu @ shared.detach().T) 

        L_joint = logq.diagonal().mean()
        mask = ~torch.eye(B, dtype=torch.bool, device=priv.device)
        L_product = logq[mask].mean()
        return L_joint - L_product

    def compute_predictor_loss(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats = {'vp': params['norm_vp'],
                 'si': params['norm_sI'],
                 'st': params['norm_sT'],
                 'tp': params['norm_tp']}
        pairs = {'vp_si': ('vp', 'si'),
                 'tp_st': ('tp', 'st'),
                 'vp_st': ('vp', 'st'),
                 'tp_si': ('tp', 'si'),
                 'vp_tp': ('vp', 'tp')}

        losses = []
        for key, (xn, yn) in pairs.items():
            x = feats[xn].detach()          
            y = feats[yn].detach()
            mu = self.vclub_predictors[key](x)
            losses.append(-(mu * y).sum(-1).mean())   # L_q(pair) = −E[µ_φ(x)·y]
        return torch.stack(losses).mean()

    def compute_predictor_diagnostics(self, params: Dict[str, torch.Tensor]) -> dict:
        feats = {'vp': params['norm_vp'],
                 'si': params['norm_sI'],
                 'st': params['norm_sT'],
                 'tp': params['norm_tp']}
        pairs = {'vp_si': ('vp', 'si'),
                 'tp_st': ('tp', 'st'),
                 'vp_st': ('vp', 'st'),
                 'tp_si': ('tp', 'si'),
                 'vp_tp': ('vp', 'tp')}

        per_pair, cos_vals, norm_vals = {}, [], []
        with torch.no_grad():
            for key, (xn, yn) in pairs.items():
                mu = self.vclub_predictors[key](feats[xn])
                y = feats[yn]
                per_pair[key] = -(mu * y).sum(-1).mean().item()
                cos_vals.append((mu * y).sum(-1))
                norm_vals.append(mu.norm(dim=-1))
        return {
            'per_pair': per_pair,
            'cos_mean': torch.cat(cos_vals).mean().item(),
            'mu_norm': torch.cat(norm_vals).mean().item(),
        }


    def compute_disentangle_loss(
        self,
        params: Dict[str, torch.Tensor],
        lambda_align: float = 1.0,
        lambda_sep:   float = 1.0,
        kappa_align:  float = 1.0,
        kappa_sep:    float = None,
    ) -> Dict:
        norm_sI = params['norm_sI']
        norm_sT = params['norm_sT']
        norm_vp = params['norm_vp']
        norm_tp = params['norm_tp']
        kappa_sep = self.kappa_sep if kappa_sep is None else kappa_sep

        L_align = self._vmf_align_loss(norm_sI, norm_sT, kappa=kappa_align)

        L_sep = (
            self._vclub_pair_loss(norm_vp, norm_sI, self.vclub_predictors['vp_si'], kappa=kappa_sep)
            + self._vclub_pair_loss(norm_tp, norm_sT, self.vclub_predictors['tp_st'], kappa=kappa_sep)
            + self._vclub_pair_loss(norm_vp, norm_sT, self.vclub_predictors['vp_st'], kappa=kappa_sep)
            + self._vclub_pair_loss(norm_tp, norm_sI, self.vclub_predictors['tp_si'], kappa=kappa_sep)
            + self._vclub_pair_loss(norm_vp, norm_tp, self.vclub_predictors['vp_tp'], kappa=kappa_sep)
        )

        total = lambda_align * L_align + lambda_sep * L_sep

        return {
            'total_loss':  total,
            'align_loss':  L_align.detach().item(),
            'sep_loss':    L_sep.detach().item(),
        }


    def compute_reconstruction_loss(
        self,
        params: Dict[str, torch.Tensor],
        cos_weight: float = 0.5,
    ) -> Dict:
        v_raw = params['v_raw'].detach()
        t_raw = params['t_raw'].detach()
        h_sI  = params['v_shared']
        h_sT  = params['t_shared']
        h_vp  = params['v_priv']
        h_tp  = params['t_priv']

        recon_v = self.visual_recon_decoder(h_vp, h_sI)
        recon_t = self.text_recon_decoder(h_tp, h_sT)

        loss_v = (
            F.smooth_l1_loss(recon_v, v_raw)
            + cos_weight * (1 - F.cosine_similarity(recon_v, v_raw, dim=-1)).mean()
        )
        loss_t = (
            F.smooth_l1_loss(recon_t, t_raw)
            + cos_weight * (1 - F.cosine_similarity(recon_t, t_raw, dim=-1)).mean()
        )
        total = loss_v + loss_t

        return {
            'total_loss': total,
            'recon_v_loss': loss_v.detach().item(),
            'recon_t_loss': loss_t.detach().item(),
        }


    def _encode_and_disentangle(
        self,
        pixel_values: torch.Tensor,
        num_images: Optional[List[int]],
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        encode_kwargs: dict,
    ):
        device = pixel_values.device if pixel_values is not None else text_input_ids.device

        if text_input_ids.size(1) > self.max_text_tokens:
            text_input_ids      = text_input_ids[:,      :self.max_text_tokens]
            text_attention_mask = text_attention_mask[:, :self.max_text_tokens]

        if 'cached_v_raw' in encode_kwargs:
            v_raw = encode_kwargs.pop('cached_v_raw')
            v_mask = encode_kwargs.pop('cached_v_mask')
        else:
            v_raw, v_mask = self.backbone.encode_images(
                pixel_values, num_images=num_images, **encode_kwargs
            )
        t_raw = self.backbone.encode_text(
            text_input_ids, text_attention_mask, self.n_text_layers,
        )
        t_raw = t_raw.to(v_raw.dtype)

        v_kpm = ~v_mask.bool()
        t_kpm = ~text_attention_mask.bool()
        h_sI = self.shared_dual.attend(v_raw, "visual", key_padding_mask=v_kpm)
        h_sT = self.shared_dual.attend(t_raw, "text",   key_padding_mask=t_kpm)
        h_vp = self.visual_private(v_raw, v_mask, h_sI)
        h_tp = self.text_private(t_raw, text_attention_mask, h_sT)

        B   = h_vp.size(0)
        N_v = h_vp.size(1)
        N_t = h_tp.size(1)
        n_q = self.n_queries

        soft_pre_seg = torch.cat([h_vp, h_sI, h_sT, h_tp], dim=1)
        seg_ids = torch.cat([
            torch.full((B, N_v), SEG_V_PRIV, dtype=torch.long, device=device),
            torch.full((B, n_q), SEG_S_I,    dtype=torch.long, device=device),
            torch.full((B, n_q), SEG_S_T,    dtype=torch.long, device=device),
            torch.full((B, N_t), SEG_T_PRIV, dtype=torch.long, device=device),
        ], dim=1)
        seg_emb = self.segment_embedding(seg_ids)
        soft_prompt = soft_pre_seg + seg_emb

        ones_nq = torch.ones(B, n_q, dtype=torch.long, device=device)
        soft_mask = torch.cat(
            [v_mask.to(torch.long), ones_nq, ones_nq, text_attention_mask.to(torch.long)],
            dim=1,
        )

        norm_vp = self._pool_and_normalize(h_vp, v_mask)
        norm_sI = self._pool_and_normalize(h_sI, ones_nq)
        norm_sT = self._pool_and_normalize(h_sT, ones_nq)
        norm_tp = self._pool_and_normalize(h_tp, text_attention_mask)

        return soft_prompt, soft_mask, {
            'norm_vp': norm_vp,
            'norm_sI': norm_sI,
            'norm_sT': norm_sT,
            'norm_tp': norm_tp,
            'v_raw': v_raw,
            't_raw': t_raw,
            'v_shared': h_sI,
            't_shared': h_sT,
            'v_priv': h_vp,
            't_priv': h_tp,
            'v_mask': v_mask,
            't_mask': text_attention_mask,
        }


    def _compact_and_pad(
        self,
        segments_embeds: list,
        segments_masks: list,
        segments_labels: list = None,
        left_pad: bool = False,
    ):
        B = segments_embeds[0].size(0)
        H = segments_embeds[0].size(-1)
        device = segments_embeds[0].device
        dtype = segments_embeds[0].dtype

        per_sample_embeds = []
        per_sample_labels = []

        for i in range(B):
            parts = []
            label_parts = []
            for seg_idx in range(len(segments_embeds)):
                mask = segments_masks[seg_idx]
                real_len = int(mask[i].sum().item())
                parts.append(segments_embeds[seg_idx][i, :real_len])
                if segments_labels is not None:
                    label_parts.append(segments_labels[seg_idx][i, :real_len])
            per_sample_embeds.append(torch.cat(parts, dim=0))
            if segments_labels is not None:
                per_sample_labels.append(torch.cat(label_parts, dim=0))

        max_len = max(e.size(0) for e in per_sample_embeds)

        padded_embeds = torch.zeros(B, max_len, H, dtype=dtype, device=device)
        padded_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        padded_labels = None
        if segments_labels is not None:
            padded_labels = torch.full(
                (B, max_len), -100, dtype=torch.long, device=device
            )

        for i in range(B):
            L = per_sample_embeds[i].size(0)
            offset = (max_len - L) if left_pad else 0
            padded_embeds[i, offset:offset + L] = per_sample_embeds[i]
            padded_mask[i, offset:offset + L] = 1
            if padded_labels is not None:
                padded_labels[i, offset:offset + L] = per_sample_labels[i]

        return padded_embeds, padded_mask, padded_labels


    def train_forward(
        self,
        pixel_values: torch.Tensor,
        num_images: Optional[List[int]],
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        system_input_ids: torch.Tensor,
        system_attention_mask: torch.Tensor,
        constraint_input_ids: torch.Tensor,
        constraint_attention_mask: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        encode_kwargs: Optional[dict] = None,
        gen_loss_weight: float = 1.0,
        disentangle_loss_weight: float = 0.1,
        recon_loss_weight: float = 0.1,
        lambda_align: float = 1.0,
        lambda_sep:   float = 1.0,
        kappa_align:  float = 1.0,
    ) -> Dict:
        device = pixel_values.device if pixel_values is not None else text_input_ids.device
        B = len(num_images)
        encode_kwargs = encode_kwargs or {}

        soft_prompt, soft_mask, params = self._encode_and_disentangle(
            pixel_values, num_images,
            text_input_ids, text_attention_mask,
            encode_kwargs,
        )
        soft_prompt_len = soft_prompt.size(1)

        system_embeds     = self.backbone.get_text_embeddings(system_input_ids)
        constraint_embeds = self.backbone.get_text_embeddings(constraint_input_ids)
        target_embeds     = self.backbone.get_text_embeddings(target_input_ids)

        llm_dtype = system_embeds.dtype

        sys_labels = torch.full(
            (B, system_input_ids.size(1)), -100, dtype=torch.long, device=device
        )
        soft_labels = torch.full(
            (B, soft_prompt_len), -100, dtype=torch.long, device=device
        )
        con_labels = torch.full(
            (B, constraint_input_ids.size(1)), -100, dtype=torch.long, device=device
        )
        target_labels = target_input_ids.clone()
        pad_id = self.backbone.get_tokenizer().pad_token_id
        if pad_id is not None:
            target_labels[target_labels == pad_id] = -100

        full_embeds, full_mask, full_labels = self._compact_and_pad(
            [system_embeds, soft_prompt.to(llm_dtype), constraint_embeds, target_embeds],
            [system_attention_mask, soft_mask,
             constraint_attention_mask, target_attention_mask],
            [sys_labels, soft_labels, con_labels, target_labels],
            left_pad=False,
        )

        gen_out = self.backbone.forward(full_embeds, full_mask, full_labels)
        generation_loss = gen_out['loss']

        dis_loss_dict = self.compute_disentangle_loss(
            params=params,
            lambda_align=lambda_align,
            lambda_sep=lambda_sep,
            kappa_align=kappa_align,
        )
        disentangle_loss = dis_loss_dict['total_loss']

        recon_loss_dict = self.compute_reconstruction_loss(params=params)
        reconstruction_loss = recon_loss_dict['total_loss']

        total_loss = (
            gen_loss_weight * generation_loss.float()
            + disentangle_loss_weight * disentangle_loss
            + recon_loss_weight * reconstruction_loss.float()
        )

        return {
            'loss':             total_loss,
            'generation_loss':  generation_loss.item(),
            'disentangle_loss': disentangle_loss.item() if torch.is_tensor(disentangle_loss) else float(disentangle_loss),
            'recon_loss':       reconstruction_loss.item() if torch.is_tensor(reconstruction_loss) else float(reconstruction_loss),
            'align_loss':       dis_loss_dict['align_loss'],
            'sep_loss':         dis_loss_dict['sep_loss'],
            'norm_vp':          params['norm_vp'],
            'norm_sI':          params['norm_sI'],
            'norm_sT':          params['norm_sT'],
            'norm_tp':          params['norm_tp'],
        }


    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        num_images: Optional[List[int]],
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        system_input_ids: torch.Tensor,
        system_attention_mask: torch.Tensor,
        constraint_input_ids: torch.Tensor,
        constraint_attention_mask: torch.Tensor,
        encode_kwargs: Optional[dict] = None,
        max_new_tokens: int = 256,
        num_beams: int = 1,
        do_sample: bool = False,
        **kwargs,
    ) -> List[str]:
        encode_kwargs = encode_kwargs or {}

        soft_prompt, soft_mask, _ = self._encode_and_disentangle(
            pixel_values, num_images,
            text_input_ids, text_attention_mask,
            encode_kwargs,
        )

        system_embeds     = self.backbone.get_text_embeddings(system_input_ids)
        constraint_embeds = self.backbone.get_text_embeddings(constraint_input_ids)

        llm_dtype = system_embeds.dtype

        full_embeds, full_mask, _ = self._compact_and_pad(
            [system_embeds, soft_prompt.to(llm_dtype), constraint_embeds],
            [system_attention_mask, soft_mask, constraint_attention_mask],
            left_pad=True,
        )

        return self.backbone.generate(
            full_embeds, full_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            **kwargs,
        )


    def generate_with_log_probs(
        self,
        pixel_values: torch.Tensor,
        num_images: Optional[List[int]],
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        system_input_ids: torch.Tensor,
        system_attention_mask: torch.Tensor,
        constraint_input_ids: torch.Tensor,
        constraint_attention_mask: torch.Tensor,
        encode_kwargs: Optional[dict] = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> Dict:
        encode_kwargs = encode_kwargs or {}

        soft_prompt, soft_mask, dis_out = self._encode_and_disentangle(
            pixel_values, num_images,
            text_input_ids, text_attention_mask,
            encode_kwargs,
        )

        system_embeds = self.backbone.get_text_embeddings(system_input_ids)
        constraint_embeds = self.backbone.get_text_embeddings(constraint_input_ids)

        llm_dtype = system_embeds.dtype
        prefix_embeds, prefix_mask, _ = self._compact_and_pad(
            [system_embeds, soft_prompt.to(llm_dtype), constraint_embeds],
            [system_attention_mask, soft_mask, constraint_attention_mask],
            left_pad=True,
        )

        generated_ids, texts = self.backbone.generate_with_ids(
            prefix_embeds, prefix_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )

        B, gen_len = generated_ids.shape
        gen_embeds = self.backbone.get_text_embeddings(generated_ids)
        full_embeds = torch.cat([prefix_embeds, gen_embeds], dim=1)

        tokenizer = self.backbone.get_tokenizer()
        pad_id = tokenizer.pad_token_id or 0
        eos_id = tokenizer.eos_token_id
        gen_attn = ((generated_ids != pad_id) & (generated_ids != eos_id)).long()
        full_mask = torch.cat([prefix_mask, gen_attn], dim=1)

        logits = self.backbone.forward_logits(full_embeds, full_mask)
        prefix_len = prefix_embeds.size(1)
        gen_logits = logits[:, prefix_len - 1: prefix_len - 1 + gen_len, :]

        log_probs = torch.log_softmax(gen_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=generated_ids.unsqueeze(-1)
        ).squeeze(-1)
        token_log_probs = token_log_probs * gen_attn.float()

        return {
            'generated_ids': generated_ids,
            'token_log_probs': token_log_probs,
            'attention_mask': gen_attn,
            'generated_texts': texts,
            'dis_out': dis_out,
        }

    def compute_log_probs(
        self,
        pixel_values: torch.Tensor,
        num_images: Optional[List[int]],
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        system_input_ids: torch.Tensor,
        system_attention_mask: torch.Tensor,
        constraint_input_ids: torch.Tensor,
        constraint_attention_mask: torch.Tensor,
        generated_ids: torch.Tensor,
        encode_kwargs: Optional[dict] = None,
    ) -> Dict:
        encode_kwargs = encode_kwargs or {}

        soft_prompt, soft_mask, _ = self._encode_and_disentangle(
            pixel_values, num_images,
            text_input_ids, text_attention_mask,
            encode_kwargs,
        )

        system_embeds = self.backbone.get_text_embeddings(system_input_ids)
        constraint_embeds = self.backbone.get_text_embeddings(constraint_input_ids)

        llm_dtype = system_embeds.dtype
        prefix_embeds, prefix_mask, _ = self._compact_and_pad(
            [system_embeds, soft_prompt.to(llm_dtype), constraint_embeds],
            [system_attention_mask, soft_mask, constraint_attention_mask],
            left_pad=True,
        )

        gen_embeds = self.backbone.get_text_embeddings(generated_ids)
        full_embeds = torch.cat([prefix_embeds, gen_embeds], dim=1)

        tokenizer = self.backbone.get_tokenizer()
        pad_id = tokenizer.pad_token_id or 0
        eos_id = tokenizer.eos_token_id
        gen_attn = ((generated_ids != pad_id) & (generated_ids != eos_id)).long()
        full_mask = torch.cat([prefix_mask, gen_attn], dim=1)

        logits = self.backbone.forward_logits(full_embeds, full_mask)
        prefix_len = prefix_embeds.size(1)
        gen_len = generated_ids.size(1)
        gen_logits = logits[:, prefix_len - 1: prefix_len - 1 + gen_len, :]

        log_probs = torch.log_softmax(gen_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=generated_ids.unsqueeze(-1)
        ).squeeze(-1)
        token_log_probs = token_log_probs * gen_attn.float()

        probs = torch.softmax(gen_logits, dim=-1)
        token_entropy = -(probs * log_probs).sum(dim=-1) * gen_attn.float()

        return {
            'token_log_probs': token_log_probs,
            'attention_mask': gen_attn,
            'token_entropy': token_entropy,
        }

    def generate_from_cached_prompt(
        self,
        soft_prompt: torch.Tensor,
        soft_mask: torch.Tensor,
        system_input_ids: torch.Tensor,
        system_attention_mask: torch.Tensor,
        constraint_input_ids: torch.Tensor,
        constraint_attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> Dict:
        system_embeds = self.backbone.get_text_embeddings(system_input_ids)
        constraint_embeds = self.backbone.get_text_embeddings(constraint_input_ids)

        llm_dtype = system_embeds.dtype
        prefix_embeds, prefix_mask, _ = self._compact_and_pad(
            [system_embeds, soft_prompt.to(llm_dtype), constraint_embeds],
            [system_attention_mask, soft_mask, constraint_attention_mask],
            left_pad=True,
        )

        generated_ids, texts = self.backbone.generate_with_ids(
            prefix_embeds, prefix_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )

        B, gen_len = generated_ids.shape
        gen_embeds = self.backbone.get_text_embeddings(generated_ids)
        full_embeds = torch.cat([prefix_embeds, gen_embeds], dim=1)

        tokenizer = self.backbone.get_tokenizer()
        pad_id = tokenizer.pad_token_id or 0
        eos_id = tokenizer.eos_token_id
        gen_attn = ((generated_ids != pad_id) & (generated_ids != eos_id)).long()
        full_mask = torch.cat([prefix_mask, gen_attn], dim=1)

        logits = self.backbone.forward_logits(full_embeds, full_mask)
        prefix_len = prefix_embeds.size(1)
        gen_logits = logits[:, prefix_len - 1: prefix_len - 1 + gen_len, :]

        log_probs = torch.log_softmax(gen_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=generated_ids.unsqueeze(-1)
        ).squeeze(-1)
        token_log_probs = token_log_probs * gen_attn.float()

        return {
            'generated_ids': generated_ids,
            'token_log_probs': token_log_probs,
            'attention_mask': gen_attn,
            'generated_texts': texts,
        }

    def compute_log_probs_from_cached_prompt(
        self,
        soft_prompt: torch.Tensor,
        soft_mask: torch.Tensor,
        system_input_ids: torch.Tensor,
        system_attention_mask: torch.Tensor,
        constraint_input_ids: torch.Tensor,
        constraint_attention_mask: torch.Tensor,
        generated_ids: torch.Tensor,
    ) -> Dict:
        system_embeds = self.backbone.get_text_embeddings(system_input_ids)
        constraint_embeds = self.backbone.get_text_embeddings(constraint_input_ids)

        llm_dtype = system_embeds.dtype
        prefix_embeds, prefix_mask, _ = self._compact_and_pad(
            [system_embeds, soft_prompt.to(llm_dtype), constraint_embeds],
            [system_attention_mask, soft_mask, constraint_attention_mask],
            left_pad=True,
        )

        gen_embeds = self.backbone.get_text_embeddings(generated_ids)
        full_embeds = torch.cat([prefix_embeds, gen_embeds], dim=1)

        tokenizer = self.backbone.get_tokenizer()
        pad_id = tokenizer.pad_token_id or 0
        eos_id = tokenizer.eos_token_id
        gen_attn = ((generated_ids != pad_id) & (generated_ids != eos_id)).long()
        full_mask = torch.cat([prefix_mask, gen_attn], dim=1)

        logits = self.backbone.forward_logits(full_embeds, full_mask)
        prefix_len = prefix_embeds.size(1)
        gen_len = generated_ids.size(1)
        gen_logits = logits[:, prefix_len - 1: prefix_len - 1 + gen_len, :]

        log_probs = torch.log_softmax(gen_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=generated_ids.unsqueeze(-1)
        ).squeeze(-1)
        token_log_probs = token_log_probs * gen_attn.float()

        probs = torch.softmax(gen_logits, dim=-1)
        token_entropy = -(probs * log_probs).sum(dim=-1) * gen_attn.float()

        return {
            'token_log_probs': token_log_probs,
            'attention_mask': gen_attn,
            'token_entropy': token_entropy,
        }
