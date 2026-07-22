from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_positions(length: int, dim: int, device, dtype):
    """Allocation-light absolute positions; attention itself stays mask-free."""
    half = dim // 2
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    scale = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / max(half - 1, 1))
    )
    value = position * scale[None]
    encoded = torch.cat((value.sin(), value.cos()), dim=-1)
    if encoded.shape[-1] < dim:
        encoded = F.pad(encoded, (0, dim - encoded.shape[-1]))
    return encoded.to(dtype=dtype)


class SDPAttention(nn.Module):
    """Self-attention that directly reaches PyTorch's fused SDPA kernels."""

    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        if dim % heads:
            raise ValueError(f"attention dim {dim} must be divisible by {heads}")
        self.heads = int(heads)
        self.head_dim = int(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = float(dropout)

    def forward(self, value, *, causal=False):
        lead, length, dim = value.shape[:-2], value.shape[-2], value.shape[-1]
        flat = value.reshape(-1, length, dim)
        q, k, v = self.qkv(flat).chunk(3, dim=-1)
        q, k, v = (
            item.reshape(flat.shape[0], length, self.heads, self.head_dim)
            .transpose(1, 2)
            for item in (q, k, v)
        )
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=bool(causal),
        )
        attended = attended.transpose(1, 2).reshape(flat.shape)
        return self.out(attended).reshape(*lead, length, dim)


class SDPCrossAttention(nn.Module):
    """Fixed-width cross-attention without a padding mask, preserving flash SDPA."""

    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        if dim % heads:
            raise ValueError(f"attention dim {dim} must be divisible by {heads}")
        self.heads = int(heads)
        self.head_dim = int(dim // heads)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = float(dropout)

    def forward(self, query, memory):
        batch, nq, dim = query.shape
        nk = memory.shape[1]
        q = self.q(query).reshape(batch, nq, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(memory).chunk(2, dim=-1)
        k = k.reshape(batch, nk, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, nk, self.heads, self.head_dim).transpose(1, 2)
        value = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        value = value.transpose(1, 2).reshape(batch, nq, dim)
        return self.out(value)


class SwiGLU(nn.Module):
    def __init__(self, dim, ratio=4.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * ratio * 2 / 3)
        hidden = max(64, ((hidden + 63) // 64) * 64)
        self.in_proj = nn.Linear(dim, 2 * hidden)
        self.out_proj = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value):
        gate, content = self.in_proj(value).chunk(2, dim=-1)
        return self.out_proj(self.dropout(F.silu(gate) * content))


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, ratio=4.0, dropout=0.0):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = SDPAttention(dim, heads, dropout)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = SwiGLU(dim, ratio, dropout)

    def forward(self, value, *, causal=False):
        value = value + self.attn(self.attn_norm(value), causal=causal)
        return value + self.ff(self.ff_norm(value))


class CrossBlock(nn.Module):
    def __init__(self, dim, heads, ratio=4.0, dropout=0.0):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.m_norm = nn.LayerNorm(dim)
        self.cross = SDPCrossAttention(dim, heads, dropout)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = SwiGLU(dim, ratio, dropout)

    def forward(self, query, memory):
        query = query + self.cross(self.q_norm(query), self.m_norm(memory))
        return query + self.ff(self.ff_norm(query))
