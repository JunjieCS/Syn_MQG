
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)



class _AsymAttnBlock(nn.Module):

    def __init__(
        self,
        qdim: int,
        kdim: int,
        out_dim: int,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0,\
            f"embed_dim({embed_dim}) text num_heads({num_heads}) text"

        self.qdim       = qdim
        self.out_dim    = out_dim
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.attn_drop  = dropout
        self.residual   = (qdim == out_dim)

        self.norm_q  = nn.LayerNorm(qdim)
        self.norm_kv = nn.LayerNorm(kdim)

        self.q_proj = nn.Linear(qdim, embed_dim)
        self.k_proj = nn.Linear(kdim, embed_dim)
        self.v_proj = nn.Linear(kdim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, out_dim)

        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N_q, _ = query.shape
        N_k = context.size(1)

        compute_dtype = self.q_proj.weight.dtype
        q = self.norm_q(query.to(self.norm_q.weight.dtype)).to(compute_dtype)
        kv = self.norm_kv(context.to(self.norm_kv.weight.dtype)).to(compute_dtype)

        q = self.q_proj(q ).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)

        if key_padding_mask is not None:
            attn_mask = (~key_padding_mask).view(B, 1, 1, N_k).expand(B, self.num_heads, N_q, N_k)
        else:
            attn_mask = None

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop if self.training else 0.0,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N_q, self.embed_dim)
        out = self.o_proj(attn_out)
        out = self.proj_drop(out)
        if self.residual:
            out = out + query.to(compute_dtype)
        return out



class _SharedDualResampler(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        n_queries: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0,\
            f"hidden_size({hidden_size}) text num_heads({num_heads}) text"

        self.n_queries = n_queries
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attn_drop = dropout

        self.queries = nn.Parameter(torch.randn(n_queries, hidden_size) * 0.02)
        self.norm_q  = nn.LayerNorm(hidden_size)
        self.q_proj  = nn.Linear(hidden_size, hidden_size)

        self.norm_kv_v  = nn.LayerNorm(hidden_size)
        self.k_proj_v   = nn.Linear(hidden_size, hidden_size)
        self.v_proj_v   = nn.Linear(hidden_size, hidden_size)
        self.o_proj_v   = nn.Linear(hidden_size, hidden_size)
        self.proj_drop_v = nn.Dropout(dropout)

        self.norm_kv_t  = nn.LayerNorm(hidden_size)
        self.k_proj_t   = nn.Linear(hidden_size, hidden_size)
        self.v_proj_t   = nn.Linear(hidden_size, hidden_size)
        self.o_proj_t   = nn.Linear(hidden_size, hidden_size)
        self.proj_drop_t = nn.Dropout(dropout)

    def attend(
        self,
        x: torch.Tensor,
        modality: str,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if modality == "visual":
            norm_kv, k_proj, v_proj, o_proj, proj_drop = (
                self.norm_kv_v, self.k_proj_v, self.v_proj_v, self.o_proj_v, self.proj_drop_v,
            )
        elif modality == "text":
            norm_kv, k_proj, v_proj, o_proj, proj_drop = (
                self.norm_kv_t, self.k_proj_t, self.v_proj_t, self.o_proj_t, self.proj_drop_t,
            )
        else:
            raise ValueError(f"modality text 'visual' text 'text'text: {modality}")

        B, N_k, _ = x.shape
        N_q = self.n_queries
        H, num_heads, head_dim = self.hidden_size, self.num_heads, self.head_dim

        query = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, N_q, H]

        compute_dtype = self.q_proj.weight.dtype
        q  = self.norm_q(query.to(self.norm_q.weight.dtype)).to(compute_dtype)
        kv = norm_kv(x.to(norm_kv.weight.dtype)).to(compute_dtype)

        q = self.q_proj(q).view(B, N_q, num_heads, head_dim).transpose(1, 2)
        k = k_proj(kv ).view(B, N_k, num_heads, head_dim).transpose(1, 2)
        v = v_proj(kv ).view(B, N_k, num_heads, head_dim).transpose(1, 2)

        if key_padding_mask is not None:
            attn_mask = (~key_padding_mask).view(B, 1, 1, N_k).expand(B, num_heads, N_q, N_k)
        else:
            attn_mask = None

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop if self.training else 0.0,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N_q, H)
        out = o_proj(attn_out)
        out = proj_drop(out)
        return out + query.to(compute_dtype)



class PrivateResampler(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        n_queries: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_queries = n_queries
        self.queries = nn.Parameter(torch.randn(n_queries, hidden_size) * 0.02)
        self.cross_attn = _AsymAttnBlock(
            qdim=hidden_size, kdim=hidden_size, out_dim=hidden_size,
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout,
        )
        logger.info(
            f"PrivateResampler: n_queries={n_queries}, hidden={hidden_size}, "
            f"num_heads={num_heads}"
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x.size(0)
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        kpm = (~attention_mask.bool()) if attention_mask is not None else None
        return self.cross_attn(queries, x, key_padding_mask=kpm)



class GatedPrivateBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = _AsymAttnBlock(
            qdim=hidden_size, kdim=hidden_size, out_dim=hidden_size,
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout,
        )
        self.cross_attn = _AsymAttnBlock(
            qdim=hidden_size, kdim=hidden_size, out_dim=hidden_size,
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout,
        )
        self.gate = nn.Parameter(torch.zeros(1))

        logger.info(
            f"GatedPrivateBlock: hidden={hidden_size}, num_heads={num_heads}"
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        shared_kv: torch.Tensor,
    ) -> torch.Tensor:
        kpm = (~attention_mask.bool()) if attention_mask is not None else None
        z_pre = self.self_attn(x, x, key_padding_mask=kpm)
        cross_out = self.cross_attn(z_pre, shared_kv, key_padding_mask=None)
        return z_pre + torch.tanh(self.gate) * cross_out


_PrivateBlock = GatedPrivateBlock



class ReconstructionDecoder(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross_attn = _AsymAttnBlock(
            qdim=hidden_size, kdim=hidden_size, out_dim=hidden_size,
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.GELU(),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Dropout(dropout),
        )
        logger.info(
            f"ReconstructionDecoder: hidden={hidden_size}, num_heads={num_heads}"
        )

    def _ffn_block(self, h: torch.Tensor) -> torch.Tensor:
        compute_dtype = self.ffn[0].weight.dtype
        return self.ffn(self.norm(h.to(self.norm.weight.dtype)).to(compute_dtype))

    def forward(
        self,
        private_feat: torch.Tensor,
        shared_feat: torch.Tensor,
    ) -> torch.Tensor:
        from torch.utils.checkpoint import checkpoint as ckpt
        h = ckpt(self.cross_attn, private_feat, shared_feat, None, use_reentrant=False)
        h = h + ckpt(self._ffn_block, h, use_reentrant=False)
        return h
