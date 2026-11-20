"""LLaMA-3-8B self-attention block, prefill (Hugging Face `LlamaAttention` eager path).

input RMSNorm -> q/k/v projections -> RoPE -> repeat_kv (GQA 32q/8kv) -> causal eager attention
(QK^T, causal mask, fp32 softmax, PV) -> o_proj -> residual add.  FP16, batch 1, 2048 tokens.
Cross-operator structure left in G0: separate q/k/v GEMMs on the same input, materialized
[B,H,S,S] scores/probabilities, repeat_kv copies, norm weight multiplied before the GEMMs.
"""

import torch
import torch.nn as nn

BATCH, SEQ, HIDDEN, N_HEADS, N_KV, HEAD_DIM = 1, 2048, 4096, 32, 8, 128
ROPE_THETA, EPS = 500000.0, 1e-5
DTYPE = torch.float16


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(x, n_rep):
    b, h, s, d = x.shape
    if n_rep == 1:
        return x
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

    def forward(self, hidden_states, cos, sin):
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

        k = repeat_kv(k, self.n_heads // self.n_kv)
        v = repeat_kv(v, self.n_heads // self.n_kv)
        scores = torch.matmul(q, k.transpose(2, 3)) * (self.head_dim ** -0.5)
        mask = torch.full((s, s), torch.finfo(DTYPE).min, dtype=DTYPE, device=h.device).triu(1)
        scores = scores + mask
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(DTYPE)
        attn = torch.matmul(probs, v).transpose(1, 2).reshape(b, s, -1)
        return residual + self.o_proj(attn)


def get_init_inputs():
    return [HIDDEN, N_HEADS, N_KV, HEAD_DIM]


def get_inputs():
    x = torch.randn(BATCH, SEQ, HIDDEN, dtype=DTYPE, device="cuda")
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, device="cuda", dtype=torch.float32) / HEAD_DIM))
    freqs = torch.outer(torch.arange(SEQ, device="cuda", dtype=torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(DTYPE).expand(BATCH, -1, -1).contiguous()
    sin = emb.sin().to(DTYPE).expand(BATCH, -1, -1).contiguous()
    return [x, cos, sin]
