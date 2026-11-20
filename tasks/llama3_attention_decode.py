"""LLaMA-3-8B self-attention block, decode step with a KV cache (Hugging Face eager + DynamicCache).

16 sequences x 1 new token, 4096 cached positions.  input RMSNorm -> q/k/v projections ->
RoPE -> append new k/v to the cache (torch.cat, as DynamicCache does) -> repeat_kv (GQA 32/8)
-> eager attention -> o_proj -> residual.  Returns the block output (the cache update itself is
owned by the caller).  Cross-operator structure in G0: the cat copies the whole cache, repeat_kv
materializes 4x expanded K/V, three projections share one input.
"""

import torch
import torch.nn as nn

BATCH, CACHE_LEN, HIDDEN, N_HEADS, N_KV, HEAD_DIM = 16, 4096, 4096, 32, 8, 128
ROPE_THETA, EPS = 500000.0, 1e-5
DTYPE = torch.float16


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(x, n_rep):
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


class Model(nn.Module):
    def __init__(self, hidden: int, n_heads: int, n_kv: int, head_dim: int):
        super().__init__()
        self.n_heads, self.n_kv, self.head_dim = n_heads, n_kv, head_dim
        self.norm_weight = nn.Parameter(1 + 0.02 * torch.randn(hidden, dtype=DTYPE))
        self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False, dtype=DTYPE)
        self.k_proj = nn.Linear(hidden, n_kv * head_dim, bias=False, dtype=DTYPE)
        self.v_proj = nn.Linear(hidden, n_kv * head_dim, bias=False, dtype=DTYPE)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False, dtype=DTYPE)

    def forward(self, hidden_states, cos, sin, k_cache, v_cache):
        b, s, _ = hidden_states.shape
        residual = hidden_states
        h = hidden_states.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS)
        h = self.norm_weight * h.to(DTYPE)

        q = self.q_proj(h).view(b, s, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, s, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, s, -1, self.head_dim).transpose(1, 2)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        k = torch.cat([k_cache, k], dim=2)
        v = torch.cat([v_cache, v], dim=2)
        k = repeat_kv(k, self.n_heads // self.n_kv)
        v = repeat_kv(v, self.n_heads // self.n_kv)
        scores = torch.matmul(q, k.transpose(2, 3)) * (self.head_dim ** -0.5)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(DTYPE)
        attn = torch.matmul(probs, v).transpose(1, 2).reshape(b, s, -1)
        return residual + self.o_proj(attn)


def get_init_inputs():
    return [HIDDEN, N_HEADS, N_KV, HEAD_DIM]


def get_inputs():
    x = torch.randn(BATCH, 1, HIDDEN, dtype=DTYPE, device="cuda")
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, device="cuda", dtype=torch.float32) / HEAD_DIM))
    freqs = torch.full((1,), float(CACHE_LEN), device="cuda")[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(DTYPE).expand(BATCH, -1, -1).contiguous()
    sin = emb.sin().to(DTYPE).expand(BATCH, -1, -1).contiguous()
    k_cache = torch.randn(BATCH, N_KV, CACHE_LEN, HEAD_DIM, dtype=DTYPE, device="cuda")
    v_cache = torch.randn(BATCH, N_KV, CACHE_LEN, HEAD_DIM, dtype=DTYPE, device="cuda")
    return [x, cos, sin, k_cache, v_cache]
