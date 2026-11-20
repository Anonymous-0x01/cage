"""Gemma-2 attention, G*: soft-capped causal GQA attention as one flex_attention call + merged QKV GEMM.

The eager path materializes [B, H, S, S] scores (bf16), soft-caps them (/50, tanh, *50), adds a dense
causal mask and runs an fp32 softmax over 4x repeated K/V.  The same math is
softmax(mask(tanh(scale * q.k / 50) * 50)) . v, i.e. a score_mod over a causal block mask: one fused
kernel, upper-triangular blocks skipped, no repeat_kv, no [S, S] tensors.
"""

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

QUERY_PRE_ATTN_SCALAR, SOFTCAP = 256, 50.0
DTYPE = torch.bfloat16


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def softcap(score, b, h, q_idx, kv_idx):
    return torch.tanh(score / SOFTCAP) * SOFTCAP


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


class Model(nn.Module):
    def __init__(self, hidden: int, n_heads: int, n_kv: int, head_dim: int):
        super().__init__()
        self.n_heads, self.n_kv, self.head_dim = n_heads, n_kv, head_dim
        self.scaling = QUERY_PRE_ATTN_SCALAR ** -0.5
        self.qkv_proj = nn.Linear(hidden, (n_heads + 2 * n_kv) * head_dim, bias=False, dtype=DTYPE)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False, dtype=DTYPE)

    def forward(self, hidden_states, cos, sin):
        b, s, _ = hidden_states.shape
        q, k, v = self.qkv_proj(hidden_states).view(b, s, -1, self.head_dim).split(
            [self.n_heads, self.n_kv, self.n_kv], dim=2)
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
        q = ((q * cos) + (rotate_half(q) * sin)).transpose(1, 2)
        k = ((k * cos) + (rotate_half(k) * sin)).transpose(1, 2)
        v = v.transpose(1, 2)
        block_mask = create_block_mask(causal, None, None, s, s, device=hidden_states.device)
        out = flex_attention(q, k, v, score_mod=softcap, block_mask=block_mask, scale=self.scaling, enable_gqa=True)
        return self.o_proj(out.transpose(1, 2).reshape(b, s, -1))


def convert_weights(sd):
    sd = dict(sd)
    sd["qkv_proj.weight"] = torch.cat([sd.pop("q_proj.weight"), sd.pop("k_proj.weight"), sd.pop("v_proj.weight")], 0)
    return sd
COMPILE_MODE = "max-autotune-no-cudagraphs"
