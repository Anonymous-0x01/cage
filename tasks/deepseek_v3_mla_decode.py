"""DeepSeek-V3 Multi-head Latent Attention (MLA), decode step over a latent KV cache.

Dimensions follow DeepSeek-V3 (hidden 7168, 128 heads, q_lora_rank 1536, kv_lora_rank 512,
qk_nope 128, qk_rope 64, v_head 128).  8 sequences x 1 new token, 2048 cached positions.
The cache holds the normalized latent c_kv [B,T,512] and the rotated shared key k_pe [B,T,64].

G0 is the "naive" (decompress-then-attend) formulation: every step it decompresses the whole
latent cache with kv_b_proj into per-head K_nope and V ([B,T,128,128] each) and runs standard
attention.  The absorbed formulation used by DeepSeek's own inference code folds W_UK into the
query and W_UV into the output, attending directly over the 512-d latents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH, CACHE_LEN, HIDDEN, N_HEADS = 8, 2048, 7168, 128
Q_RANK, KV_RANK, NOPE, ROPE, VDIM = 1536, 512, 128, 64, 128
EPS = 1e-6
DTYPE = torch.float16


def rms_norm(x, w, eps=EPS):
    xf = x.to(torch.float32)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class Model(nn.Module):
    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.input_norm = nn.Parameter(1 + 0.02 * torch.randn(hidden, dtype=DTYPE))
        self.q_a_proj = nn.Linear(hidden, Q_RANK, bias=False, dtype=DTYPE)
        self.q_a_norm = nn.Parameter(1 + 0.02 * torch.randn(Q_RANK, dtype=DTYPE))
        self.q_b_proj = nn.Linear(Q_RANK, n_heads * (NOPE + ROPE), bias=False, dtype=DTYPE)
        self.kv_a_proj = nn.Linear(hidden, KV_RANK + ROPE, bias=False, dtype=DTYPE)
        self.kv_a_norm = nn.Parameter(1 + 0.02 * torch.randn(KV_RANK, dtype=DTYPE))
        self.kv_b_proj = nn.Linear(KV_RANK, n_heads * (NOPE + VDIM), bias=False, dtype=DTYPE)
        self.o_proj = nn.Linear(n_heads * VDIM, hidden, bias=False, dtype=DTYPE)
        self.scale = (NOPE + ROPE) ** -0.5

    def forward(self, hidden_states, cos, sin, c_kv_cache, k_pe_cache):
        b, s, _ = hidden_states.shape
        residual = hidden_states
        h = rms_norm(hidden_states, self.input_norm)

        q = self.q_b_proj(rms_norm(self.q_a_proj(h), self.q_a_norm)).view(b, s, self.n_heads, NOPE + ROPE)
        q_nope, q_pe = q.split([NOPE, ROPE], dim=-1)
        q_pe = q_pe * cos.unsqueeze(2) + rotate_half(q_pe) * sin.unsqueeze(2)

        kv = self.kv_a_proj(h)
        c_new, k_pe_new = kv.split([KV_RANK, ROPE], dim=-1)
        c_new = rms_norm(c_new, self.kv_a_norm)
        k_pe_new = k_pe_new * cos + rotate_half(k_pe_new) * sin

        c_all = torch.cat([c_kv_cache, c_new], dim=1)                       # [B, T+1, 512]
        k_pe_all = torch.cat([k_pe_cache, k_pe_new], dim=1)                 # [B, T+1, 64]
        kv_full = self.kv_b_proj(c_all).view(b, -1, self.n_heads, NOPE + VDIM)
        k_nope, v = kv_full.split([NOPE, VDIM], dim=-1)                     # [B, T+1, H, 128] each
        k = torch.cat([k_nope, k_pe_all.unsqueeze(2).expand(-1, -1, self.n_heads, -1)], dim=-1)
        qf = torch.cat([q_nope, q_pe], dim=-1)

        scores = torch.einsum("bshd,bthd->bhst", qf, k) * self.scale
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(DTYPE)
        out = torch.einsum("bhst,bthd->bshd", probs, v).reshape(b, s, -1)
        return residual + self.o_proj(out)


def get_init_inputs():
    return [HIDDEN, N_HEADS]


def get_inputs():
    x = torch.randn(BATCH, 1, HIDDEN, dtype=DTYPE, device="cuda")
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, ROPE, 2, device="cuda", dtype=torch.float32) / ROPE))
    freqs = float(CACHE_LEN) * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(DTYPE).view(1, 1, ROPE).expand(BATCH, 1, ROPE).contiguous()
    sin = emb.sin().to(DTYPE).view(1, 1, ROPE).expand(BATCH, 1, ROPE).contiguous()
    c_kv_cache = torch.randn(BATCH, CACHE_LEN, KV_RANK, dtype=DTYPE, device="cuda")
    k_pe_cache = torch.randn(BATCH, CACHE_LEN, ROPE, dtype=DTYPE, device="cuda")
    return [x, cos, sin, c_kv_cache, k_pe_cache]
