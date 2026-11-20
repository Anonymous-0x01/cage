"""LLaMA-3 prefill attention, G*: fused causal GQA attention + merged QKV projection.

QK^T * scale + causal mask + fp32 softmax + PV with repeat_kv is exactly causal scaled dot-product
attention with grouped KV heads, so it becomes one flash-attention call that never materializes
the [B, H, S, S] scores or the 4x expanded K/V.  q/k/v share one input -> one GEMM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-5
DTYPE = torch.float16


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class Model(nn.Module):
    def __init__(self, hidden: int, n_heads: int, n_kv: int, head_dim: int):
        super().__init__()
        self.n_heads, self.n_kv, self.head_dim = n_heads, n_kv, head_dim
        self.norm_weight = nn.Parameter(torch.empty(hidden, dtype=DTYPE))
        self.qkv_proj = nn.Linear(hidden, (n_heads + 2 * n_kv) * head_dim, bias=False, dtype=DTYPE)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False, dtype=DTYPE)

    def forward(self, hidden_states, cos, sin):
        b, s, _ = hidden_states.shape
        nq, nkv, d = self.n_heads, self.n_kv, self.head_dim
        residual = hidden_states
        h = hidden_states.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS)
        h = self.norm_weight * h.to(DTYPE)

        q, k, v = self.qkv_proj(h).view(b, s, nq + 2 * nkv, d).split([nq, nkv, nkv], dim=2)
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
        q = ((q * cos) + (rotate_half(q) * sin)).transpose(1, 2)
        k = ((k * cos) + (rotate_half(k) * sin)).transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return residual + self.o_proj(attn.transpose(1, 2).reshape(b, s, nq * d))


def convert_weights(sd):
    sd = dict(sd)
    sd["qkv_proj.weight"] = torch.cat([sd.pop("q_proj.weight"), sd.pop("k_proj.weight"), sd.pop("v_proj.weight")], 0)
    return sd
