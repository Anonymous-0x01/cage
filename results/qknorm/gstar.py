"""QK-Norm + RoPE before attention (Qwen3-style), FP16, batch 8 x seq 1024, 32 q heads / 8 kv heads, head_dim 128.

Takes the fused QKV projection output, splits it into q/k/v, applies per-head RMSNorm to
q and k, applies rotary position embedding, and returns q/k/v in [B, H, S, D] layout.
"""

import torch
import torch.nn as nn

BATCH, SEQ = 8, 1024
N_Q, N_KV, HEAD_DIM = 32, 8, 128
DTYPE = torch.float16


def rms_norm(x, weight, eps):
    xf = x.to(torch.float32)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Model(nn.Module):
    def __init__(self, n_q: int, n_kv: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.n_q, self.n_kv, self.head_dim, self.eps = n_q, n_kv, head_dim, eps
        self.q_norm_weight = nn.Parameter(1 + 0.01 * torch.randn(head_dim, dtype=DTYPE))
        self.k_norm_weight = nn.Parameter(1 + 0.01 * torch.randn(head_dim, dtype=DTYPE))

    def forward(self, qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        b, s, _ = qkv.shape
        q_size, kv_size = self.n_q * self.head_dim, self.n_kv * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.reshape(b, s, self.n_q, self.head_dim)
        k = k.reshape(b, s, self.n_kv, self.head_dim)
        v = v.reshape(b, s, self.n_kv, self.head_dim)

        q = rms_norm(q, self.q_norm_weight, self.eps).transpose(1, 2)
        k = rms_norm(k, self.k_norm_weight, self.eps).transpose(1, 2)
        v = v.transpose(1, 2).contiguous()

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        return q.contiguous(), k.contiguous(), v


def get_init_inputs():
    return [N_Q, N_KV, HEAD_DIM]


def get_inputs():
    qkv = torch.randn(BATCH, SEQ, (N_Q + 2 * N_KV) * HEAD_DIM, dtype=DTYPE, device="cuda")
    pos = torch.arange(SEQ, device="cuda", dtype=torch.float32)
    inv_freq = 1.0 / (1e6 ** (torch.arange(0, HEAD_DIM, 2, device="cuda", dtype=torch.float32) / HEAD_DIM))
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(DTYPE).unsqueeze(0).expand(BATCH, -1, -1).contiguous()
    sin = emb.sin().to(DTYPE).unsqueeze(0).expand(BATCH, -1, -1).contiguous()
    return [qkv, cos, sin]
INDUCTOR_CONFIG = {"coordinate_descent_tuning": True}
