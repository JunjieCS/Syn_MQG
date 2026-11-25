
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
import logging

from .mllm_backbone import MLLMBackbone
from .early_disentangle_projectors import (
    GatedPrivateBlock,
    _SharedDualResampler,
)

logger = logging.getLogger(__name__)

SEG_V_PRIV = 0
SEG_S_I    = 1
SEG_S_T    = 2
SEG_T_PRIV = 3


class MLLMEarlyDisentangleModel(nn.Module):

    def __init__(
        self,
        backbone: MLLMBackbone,
        disentangle_config: Optional[dict] = None,
    ):
        super().__init__()

        cfg = disentangle_config or {}
        H = backbone.hidden_dim
        dropout = cfg.get('dropout', 0.1)
        num_heads = cfg.get('num_heads', 8)
        n_queries_shared = cfg.get('n_queries_shared', 32)

        self.n_text_layers = cfg.get('n_text_layers', 4)
        self.infonce_temperature = cfg.get('infonce_temperature', 0.07)
        self.dcor_detach_shared = cfg.get('dcor_detach_shared', True)
        self.max_text_tokens = cfg.get('max_text_tokens', 512)
        self.n_shared = n_queries_shared

        self.backbone = backbone

        self.shared_dual = _SharedDualResampler(
            hidden_size=H, n_queries=n_queries_shared,
            num_heads=num_heads, dropout=dropout,
        )

        self.visual_private = GatedPrivateBlock(
            hidden_size=H, num_heads=num_heads, dropout=dropout,
        )
        self.text_private = GatedPrivateBlock(
            hidden_size=H, num_heads=num_heads, dropout=dropout,
        )

        # Segment embedding
        self.segment_embedding = nn.Embedding(4, H)
        nn.init.normal_(self.segment_embedding.weight, mean=0.0, std=0.02)

        logger.info(
            f"MLLMEarlyDisentangleModel initialized: H={H}, "
            f"shared={n_queries_shared}, "
            "private=GatedPrivateBlock(variable length)"
        )


    @staticmethod
    def _masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).to(x.dtype)
        return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

    @staticmethod
    def _distance_correlation(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        X = X.float()
        Y = Y.float()
        a = torch.cdist(X, X)
        b = torch.cdist(Y, Y)
        A = a - a.mean(0, keepdim=True) - a.mean(1, keepdim=True) + a.mean()
        B = b - b.mean(0, keepdim=True) - b.mean(1, keepdim=True) + b.mean()
        dCov_XY = (A * B).mean().clamp(min=0)
        dCov_XX = (A * A).mean().clamp(min=0)
        dCov_YY = (B * B).mean().clamp(min=0)
        return dCov_XY / (dCov_XX * dCov_YY).sqrt().clamp(min=1e-8)


    def compute_disentangle_loss(
        self,
        v_priv: torch.Tensor,
        t_priv: torch.Tensor,
        v_shared: torch.Tensor,
        t_shared: torch.Tensor,
        v_mask: torch.Tensor,
        t_mask: torch.Tensor,
        lambda_sep: float = 1.0,
        lambda_orth: float = 0.5,
        temperature: float = 0.07,
    ) -> Dict:
        z_vp = F.normalize(self._masked_mean_pool(v_priv, v_mask), dim=-1)
        z_tp = F.normalize(self._masked_mean_pool(t_priv, t_mask), dim=-1)
        z_si = F.normalize(v_shared.mean(dim=1), dim=-1)
        z_st = F.normalize(t_shared.mean(dim=1), dim=-1)

        sim = torch.matmul(z_si, z_st.t()) / temperature
        labels = torch.arange(sim.size(0), device=sim.device)
        shared_loss = 0.5 * (
            F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)
        )

        priv_sep_loss = self._distance_correlation(z_vp, z_tp)

        z_si_orth = z_si.detach() if self.dcor_detach_shared else z_si
        z_st_orth = z_st.detach() if self.dcor_detach_shared else z_st
        orth_loss = (
            self._distance_correlation(z_vp, z_si_orth) +
            self._distance_correlation(z_tp, z_st_orth) +
            self._distance_correlation(z_vp, z_st_orth) +
            self._distance_correlation(z_tp, z_si_orth)
        )

        total = shared_loss + lambda_sep * priv_sep_loss + lambda_orth * orth_loss
        return {
            'total_loss': total,
            'shared_loss': shared_loss.item(),
            'priv_sep_loss': priv_sep_loss.item(),
            'orth_loss': orth_loss.item(),
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
            text_input_ids = text_input_ids[:, :self.max_text_tokens]
            text_attention_mask = text_attention_mask[:, :self.max_text_tokens]

        if 'cached_v_raw' in encode_kwargs:
            v_raw = encode_kwargs.pop('cached_v_raw')
            v_raw_mask = encode_kwargs.pop('cached_v_mask')
        else:
            v_raw, v_raw_mask = self.backbone.encode_images(
                pixel_values, num_images=num_images, **encode_kwargs
            )

        t_raw = self.backbone.encode_text(
            text_input_ids, text_attention_mask, self.n_text_layers,
        )
        t_raw = t_raw.to(v_raw.dtype)

        v_kpm = ~v_raw_mask.bool()
        t_kpm = ~text_attention_mask.bool()
        v_shared = self.shared_dual.attend(v_raw, modality="visual", key_padding_mask=v_kpm)
        t_shared = self.shared_dual.attend(t_raw, modality="text", key_padding_mask=t_kpm)

        v_priv = self.visual_private(v_raw, v_raw_mask, v_shared)
        t_priv = self.text_private(t_raw, text_attention_mask, t_shared)

        B = v_priv.size(0)
        N_v = v_priv.size(1)
        N_t = t_priv.size(1)

        soft_pre_seg = torch.cat([v_priv, v_shared, t_shared, t_priv], dim=1)

        # 6. Segment embedding
        seg_ids = torch.cat([
            torch.full((B, N_v), SEG_V_PRIV, dtype=torch.long, device=device),
            torch.full((B, self.n_shared), SEG_S_I, dtype=torch.long, device=device),
            torch.full((B, self.n_shared), SEG_S_T, dtype=torch.long, device=device),
            torch.full((B, N_t), SEG_T_PRIV, dtype=torch.long, device=device),
        ], dim=1)
        soft_prompt = soft_pre_seg + self.segment_embedding(seg_ids)

        shared_mask = torch.ones(B, 2 * self.n_shared, dtype=torch.long, device=device)
        soft_mask = torch.cat([v_raw_mask, shared_mask, text_attention_mask], dim=1)

        return soft_prompt, soft_mask, {
            'v_shared': v_shared,
            't_shared': t_shared,
            'v_priv': v_priv,
            't_priv': t_priv,
            'v_mask': v_raw_mask,
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
        lambda_sep: float = 1.0,
        lambda_orth: float = 0.5,
        temperature: float = 0.07,
    ) -> Dict:
        device = text_input_ids.device
        B = len(num_images)
        encode_kwargs = encode_kwargs or {}

        soft_prompt, soft_mask, dis_out = self._encode_and_disentangle(
            pixel_values, num_images,
            text_input_ids, text_attention_mask,
            encode_kwargs,
        )

        system_embeds = self.backbone.get_text_embeddings(system_input_ids)
        constraint_embeds = self.backbone.get_text_embeddings(constraint_input_ids)
        target_embeds = self.backbone.get_text_embeddings(target_input_ids)

        llm_dtype = system_embeds.dtype

        sys_labels = torch.full(
            (B, system_input_ids.size(1)), -100, dtype=torch.long, device=device
        )
        soft_labels = torch.full(
            (B, soft_prompt.size(1)), -100, dtype=torch.long, device=device
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
            v_priv=dis_out['v_priv'],
            t_priv=dis_out['t_priv'],
            v_shared=dis_out['v_shared'],
            t_shared=dis_out['t_shared'],
            v_mask=dis_out['v_mask'],
            t_mask=dis_out['t_mask'],
            lambda_sep=lambda_sep,
            lambda_orth=lambda_orth,
            temperature=temperature,
        )
        disentangle_loss = dis_loss_dict['total_loss']

        total_loss = gen_loss_weight * generation_loss + disentangle_loss_weight * disentangle_loss

        return {
            'loss': total_loss,
            'generation_loss': generation_loss.item(),
            'disentangle_loss': disentangle_loss.item(),
            'shared_loss': dis_loss_dict['shared_loss'],
            'priv_sep_loss': dis_loss_dict['priv_sep_loss'],
            'orth_loss': dis_loss_dict['orth_loss'],
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

        system_embeds = self.backbone.get_text_embeddings(system_input_ids)
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
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        encode_kwargs: Optional[dict] = None,
    ) -> Dict:
        encode_kwargs = encode_kwargs or {}

        soft_prompt, soft_mask, dis_out = self._encode_and_disentangle(
            pixel_values, num_images,
            text_input_ids, text_attention_mask, encode_kwargs,
        )

        system_embeds = self.backbone.get_text_embeddings(system_input_ids)
        constraint_embeds = self.backbone.get_text_embeddings(constraint_input_ids)
        llm_dtype = system_embeds.dtype

        L_sys = system_input_ids.size(1)
        prefix_embeds = torch.cat(
            [system_embeds, soft_prompt.to(llm_dtype), constraint_embeds], dim=1
        )
        prefix_mask = torch.cat(
            [system_attention_mask, soft_mask, constraint_attention_mask], dim=1
        )

        generated_ids, generated_texts = self.backbone.generate_with_ids(
            prefix_embeds, prefix_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            num_beams=1,
        )

        # behavior log probs
        B, T = generated_ids.size()
        tokenizer = self.backbone.get_tokenizer()
        pad_id = tokenizer.pad_token_id or 0
        gen_embeds = self.backbone.get_text_embeddings(generated_ids).to(llm_dtype)
        full_embeds = torch.cat([prefix_embeds, gen_embeds], dim=1)
        gen_mask = (generated_ids != pad_id).long()
        full_mask = torch.cat([prefix_mask, gen_mask], dim=1)

        logits = self.backbone.get_lm_logits(full_embeds, full_mask)
        L_pre = prefix_embeds.size(1)
        gen_logits = logits[:, L_pre - 1:L_pre - 1 + T, :].contiguous()
        del logits, full_embeds, gen_embeds
        token_log_probs = F.log_softmax(gen_logits, dim=-1).gather(
            2, generated_ids.unsqueeze(-1)
        ).squeeze(-1)
        del gen_logits

        # soft_prompt_sections
        N_v = dis_out['v_priv'].size(1)
        K_s = self.n_shared
        N_t = dis_out['t_priv'].size(1)
        soft_prompt_sections = {
            'v_priv': (0, N_v),
            's_I':    (N_v, N_v + K_s),
            's_T':    (N_v + K_s, N_v + 2 * K_s),
            't_priv': (N_v + 2 * K_s, N_v + 2 * K_s + N_t),
        }

        return {
            'generated_ids':        generated_ids,
            'token_log_probs':      token_log_probs,
            'generated_texts':      generated_texts,
            'soft_prompt':          soft_prompt,
            'prefix_embeds':        prefix_embeds,
            'prefix_mask':          prefix_mask,
            'soft_start':           L_sys,
            'soft_prompt_sections': soft_prompt_sections,
        }

    def compute_logprobs(
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
            text_input_ids, text_attention_mask, encode_kwargs,
        )

        system_embeds = self.backbone.get_text_embeddings(system_input_ids)
        constraint_embeds = self.backbone.get_text_embeddings(constraint_input_ids)
        llm_dtype = system_embeds.dtype

        prefix_embeds = torch.cat(
            [system_embeds, soft_prompt.to(llm_dtype), constraint_embeds], dim=1
        )
        prefix_mask = torch.cat(
            [system_attention_mask, soft_mask, constraint_attention_mask], dim=1
        )

        tokenizer = self.backbone.get_tokenizer()
        pad_id = tokenizer.pad_token_id or 0
        B, T = generated_ids.size()
        gen_embeds = self.backbone.get_text_embeddings(generated_ids).to(llm_dtype)
        gen_mask = (generated_ids != pad_id).long()

        full_embeds = torch.cat([prefix_embeds, gen_embeds], dim=1)
        full_mask = torch.cat([prefix_mask, gen_mask], dim=1)

        logits = self.backbone.get_lm_logits(full_embeds, full_mask)
        L_pre = prefix_embeds.size(1)
        gen_logits = logits[:, L_pre - 1:L_pre - 1 + T, :].contiguous()
        del logits, full_embeds, gen_embeds

        log_probs = F.log_softmax(gen_logits, dim=-1)
        token_log_probs = log_probs.gather(2, generated_ids.unsqueeze(-1)).squeeze(-1)
        probs = F.softmax(gen_logits, dim=-1)
        token_entropy = -(probs * log_probs).sum(dim=-1)
        del gen_logits

        return {
            'token_log_probs': token_log_probs,
            'token_entropy':   token_entropy,
            'attention_mask':  gen_mask,
        }

    def compute_ablation_logprobs(
        self,
        soft_prompt: torch.Tensor,
        prefix_embeds: torch.Tensor,
        prefix_mask: torch.Tensor,
        question_ids: torch.Tensor,
        soft_start: int,
        soft_prompt_sections: Optional[Dict] = None,
    ) -> Dict:
        llm_dtype = prefix_embeds.dtype

        secs = soft_prompt_sections or {}
        v_s, v_e = secs.get('v_priv', (0, 0))
        t_s, t_e = secs.get('t_priv', (0, 0))

        tokenizer = self.backbone.get_tokenizer()
        pad_id = tokenizer.pad_token_id or 0
        q_mask = (question_ids != pad_id).long()
        q_embeds = self.backbone.get_text_embeddings(question_ids).to(llm_dtype)
        q_mask_f = q_mask.float()
        L_pre = prefix_embeds.size(1)
        q_len = question_ids.size(1)

        ablation_cases = [
            [],
            [(v_s, v_e)],
            [(t_s, t_e)],
            [(v_s, v_e), (t_s, t_e)],
        ]
        mean_lp_list = []
        for ranges in ablation_cases:
            if ranges:
                p = prefix_embeds.clone()
                for rs, re in ranges:
                    if re > rs:
                        p[:, soft_start + rs:soft_start + re, :] = 0.0
            else:
                p = prefix_embeds
            full_e = torch.cat([p, q_embeds], dim=1)
            full_m = torch.cat([prefix_mask, q_mask], dim=1)
            if ranges:
                del p
            logits = self.backbone.get_lm_logits(full_e, full_m)
            q_logits = logits[:, L_pre - 1:L_pre - 1 + q_len, :].contiguous()
            del logits, full_e
            log_probs = F.log_softmax(q_logits, dim=-1)
            token_lp = log_probs.gather(2, question_ids.unsqueeze(-1)).squeeze(-1)
            del q_logits, log_probs
            mean_lp_list.append((token_lp * q_mask_f).sum(1) / q_mask_f.sum(1).clamp(min=1))

        lp_full, lp_no_v, lp_no_t, lp_no_vt = mean_lp_list
        syn = lp_full - lp_no_v - lp_no_t + lp_no_vt

        return {
            'syn':     syn.detach(),
            'lp_full': lp_full.detach(),
            'lp_no_v': lp_no_v.detach(),
            'lp_no_t': lp_no_t.detach(),
            'lp_no_vt': lp_no_vt.detach(),
        }
