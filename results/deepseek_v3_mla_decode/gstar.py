"""DeepSeek-V3 MLA decode, G*: weight-absorbed attention over the latent cache.

k_nope[t,h] = W_UK[h] c_t and v[t,h] = W_UV[h] c_t, so
  q_nope[h] . k_nope[t,h] = (W_UK[h]^T q_nope[h]) . c_t        (absorb W_UK into the query)
  sum_t p[h,t] v[t,h]     = W_UV[h] (sum_t p[h,t] c_t)          (absorb W_UV into the output)
Attention runs directly over the 512-d latents; the cache is never decompressed, and the new
token is handled separately instead of concatenating it onto the cache.
"""

import torch
import torch.nn as nn

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
        self.input_norm = nn.Parameter(torch.empty(hidden, dtype=DTYPE))
        self.q_a_proj = nn.Linear(hidden, Q_RANK, bias=False, dtype=DTYPE)
        self.q_a_norm = nn.Parameter(torch.empty(Q_RANK, dtype=DTYPE))
        self.q_b_proj = nn.Linear(Q_RANK, n_heads * (NOPE + ROPE), bias=False, dtype=DTYPE)
        self.kv_a_proj = nn.Linear(hidden, KV_RANK + ROPE, bias=False, dtype=DTYPE)
        self.kv_a_norm = nn.Parameter(torch.empty(KV_RANK, dtype=DTYPE))
        self.w_uk = nn.Parameter(torch.empty(n_heads, NOPE, KV_RANK, dtype=DTYPE))   # [H, 128, 512]
        self.w_uv_t = nn.Parameter(torch.empty(n_heads, KV_RANK, VDIM, dtype=DTYPE))  # [H, 512, 128]
        self.o_proj = nn.Linear(n_heads * VDIM, hidden, bias=False, dtype=DTYPE)
        self.scale = (NOPE + ROPE) ** -0.5

    def forward(self, hidden_states, cos, sin, c_kv_cache, k_pe_cache):
        b, s, _ = hidden_states.shape
        H = self.n_heads
        residual = hidden_states
        h = rms_norm(hidden_states, self.input_norm)

        q = self.q_b_proj(rms_norm(self.q_a_proj(h), self.q_a_norm)).view(b, H, NOPE + ROPE)
        q_nope, q_pe = q.split([NOPE, ROPE], dim=-1)
        q_pe = q_pe * cos + rotate_half(q_pe) * sin                                     # [B, H, 64]

        kv = self.kv_a_proj(h).view(b, KV_RANK + ROPE)
        c_new, k_pe_new = kv.split([KV_RANK, ROPE], dim=-1)
        c_new = rms_norm(c_new, self.kv_a_norm)                                          # [B, 512]
        k_pe_new = k_pe_new * cos.view(b, ROPE) + rotate_half(k_pe_new) * sin.view(b, ROPE)

        q_lat = torch.bmm(q_nope.transpose(0, 1), self.w_uk).transpose(0, 1)             # [B, H, 512]
        s_cache = torch.bmm(q_lat, c_kv_cache.transpose(1, 2)) + torch.bmm(q_pe, k_pe_cache.transpose(1, 2))
        s_new = (q_lat * c_new.unsqueeze(1)).sum(-1, keepdim=True) + (q_pe * k_pe_new.unsqueeze(1)).sum(-1, keepdim=True)
        scores = torch.cat([s_cache, s_new.to(s_cache.dtype)], dim=-1) * self.scale      # [B, H, T+1]
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(DTYPE)

        t = c_kv_cache.shape[1]
        o_lat = torch.bmm(probs[..., :t], c_kv_cache) + probs[..., t:] * c_new.unsqueeze(1)   # [B, H, 512]
        out = torch.bmm(o_lat.transpose(0, 1), self.w_uv_t).transpose(0, 1)             # [B, H, 128]
        return residual + self.o_proj(out.reshape(b, s, H * VDIM))


def convert_weights(sd):
    sd = dict(sd)
    w = sd.pop("kv_b_proj.weight")                       # [H*(128+128), 512]
    n_heads = w.shape[0] // (NOPE + VDIM)
    w = w.view(n_heads, NOPE + VDIM, KV_RANK)
    sd["w_uk"] = w[:, :NOPE, :].contiguous()
    sd["w_uv_t"] = w[:, NOPE:, :].transpose(1, 2).contiguous()
    return sd
COMPILE_MODE = "reduce-overhead"
