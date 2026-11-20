"""LLaMA-3 decode attention, G*: grouped-query attention over the cache without copies.

* repeat_kv: query head i uses KV head i // 4, so q is viewed as [B, 8, 4, D] and multiplied
  against the 8 KV heads directly (no 4x expanded K/V).
* cat(cache, new): attention over [cache ; new] = softmax over [s_cache ; s_new], so the new
  token's score/value are handled separately and the cache is read in place (no cache copy).
* q/k/v projections share one input -> one GEMM.
"""

import torch
import torch.nn as nn

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

    def forward(self, hidden_states, cos, sin, k_cache, v_cache):
        b, s, _ = hidden_states.shape
        nq, nkv, d = self.n_heads, self.n_kv, self.head_dim
        g = nq // nkv
        residual = hidden_states
        h = hidden_states.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS)
        h = self.norm_weight * h.to(DTYPE)

        q, k, v = self.qkv_proj(h).view(b, nq + 2 * nkv, d).split([nq, nkv, nkv], dim=1)
        cos, sin = cos.view(b, 1, d), sin.view(b, 1, d)
        q = (q * cos) + (rotate_half(q) * sin)                       # [B, 32, D]
        k = (k * cos) + (rotate_half(k) * sin)                       # [B, 8, D]

        qg = q.view(b, nkv, g, d)                                    # [B, 8, 4, D]
        scale = d ** -0.5
        s_cache = torch.matmul(qg, k_cache.transpose(2, 3)) * scale  # [B, 8, 4, T]
        s_new = (qg * k.unsqueeze(2)).sum(-1, keepdim=True) * scale  # [B, 8, 4, 1]
        probs = torch.softmax(torch.cat([s_cache, s_new.to(s_cache.dtype)], dim=-1), dim=-1, dtype=torch.float32).to(DTYPE)
        t = k_cache.shape[2]
        out = torch.matmul(probs[..., :t], v_cache) + probs[..., t:] * v.unsqueeze(2)   # [B, 8, 4, D]
        return residual + self.o_proj(out.reshape(b, s, nq * d))


def convert_weights(sd):
    sd = dict(sd)
    sd["qkv_proj.weight"] = torch.cat([sd.pop("q_proj.weight"), sd.pop("k_proj.weight"), sd.pop("v_proj.weight")], 0)
    return sd
