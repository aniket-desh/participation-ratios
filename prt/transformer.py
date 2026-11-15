"""Small decoder-only transformer for Stage D (experiment plan §8.1; derivations §4.1).

Learned token and positional embeddings, causal multi-head attention with 1/sqrt(d_h) scores, GELU MLP of width
mlp_mult*d, untied linear readout, no dropout. norm = "none" (baseline) or "pre" (pre-LayerNorm ablation);
mlp = False gives the attention-only ablation. Every named site is captured at every position when requested:
    L{l}.resid_pre   input to block l (embedding for l = 0)
    L{l}.ln1, L{l}.ln2   normalized inputs to attention/MLP (pre-LN only)
    L{l}.attn_out    attention branch output added to the residual
    L{l}.resid_mid   residual after attention
    L{l}.mlp_out     MLP branch output
    L{l}.resid_post  residual after the block
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab: int = 4
    ctx: int = 16
    d: int = 64
    n_heads: int = 2
    d_head: int = 32
    n_layers: int = 1
    mlp_mult: int = 4
    norm: str = "none"   # "none" | "pre"
    mlp: bool = True


class Block(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        hd = c.n_heads * c.d_head
        self.qkv = nn.Linear(c.d, 3 * hd)
        self.o = nn.Linear(hd, c.d)
        if c.mlp:
            self.fc1 = nn.Linear(c.d, c.mlp_mult * c.d)
            self.fc2 = nn.Linear(c.mlp_mult * c.d, c.d)
        if c.norm == "pre":
            self.ln1 = nn.LayerNorm(c.d)
            self.ln2 = nn.LayerNorm(c.d)

    def forward(self, h, cache: dict | None, prefix: str):
        c = self.c
        B, T, _ = h.shape
        if cache is not None:
            cache[prefix + "resid_pre"] = h
        u = self.ln1(h) if c.norm == "pre" else h
        if cache is not None and c.norm == "pre":
            cache[prefix + "ln1"] = u
        q, k, v = self.qkv(u).view(B, T, 3, c.n_heads, c.d_head).permute(2, 0, 3, 1, 4)
        z = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # scale 1/sqrt(d_head)
        a = self.o(z.transpose(1, 2).reshape(B, T, -1))
        h = h + a
        if cache is not None:
            cache[prefix + "attn_out"] = a
            cache[prefix + "resid_mid"] = h
        if c.mlp:
            y = self.ln2(h) if c.norm == "pre" else h
            if cache is not None and c.norm == "pre":
                cache[prefix + "ln2"] = y
            m = self.fc2(F.gelu(self.fc1(y)))
            h = h + m
            if cache is not None:
                cache[prefix + "mlp_out"] = m
        if cache is not None:
            cache[prefix + "resid_post"] = h
        return h


class Transformer(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d)
        self.pos = nn.Embedding(c.ctx, c.d)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.unembed = nn.Linear(c.d, c.vocab)

    def forward(self, toks, cache: dict | None = None):
        T = toks.shape[1]
        h = self.tok(toks) + self.pos(torch.arange(T, device=toks.device))
        for l, blk in enumerate(self.blocks):
            h = blk(h, cache, f"L{l}.")
        return self.unembed(h)

    def sites(self) -> list[str]:
        out = []
        for l in range(self.c.n_layers):
            out += [f"L{l}.resid_pre"] + ([f"L{l}.ln1"] if self.c.norm == "pre" else []) + [f"L{l}.attn_out", f"L{l}.resid_mid"]
            if self.c.mlp:
                out += ([f"L{l}.ln2"] if self.c.norm == "pre" else []) + [f"L{l}.mlp_out"]
            out += [f"L{l}.resid_post"]
        return out

    def config_dict(self):
        return asdict(self.c)
