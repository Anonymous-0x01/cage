"""SD VAE mid-block attention, G*: tensor-core GEMM attention for the 512-wide head + GroupNorm folded into QKV.

* heads = 1 with head_dim 512 exceeds the flash / cuDNN limits, so SDPA falls back to the sm80 memory-
  efficient kernel at ~26 TFLOPS (88% of the block).  Materializing the scores is cheaper here:
  S = Q K^T (fp16 tensor-core GEMM, fp32 output), fp32 softmax, P V (fp16 GEMM), ~69 GFLOP at GEMM speed.
* GroupNorm's affine (gamma, beta) feeds three linear layers: W (gamma * n + beta) + b =
  (W diag gamma) n + (W beta + b), so the affine is folded offline into one merged QKV weight / bias and
  the normalized activations are consumed directly in [B, HW, C] layout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

GROUPS, EPS = 32, 1e-6
DTYPE = torch.float16


class Model(nn.Module):
    def __init__(self, channels: int, heads: int, groups: int):
        super().__init__()
        self.channels, self.groups = channels, groups
        self.qkv = nn.Linear(channels, 3 * channels, bias=True, dtype=DTYPE)
        self.to_out = nn.Linear(channels, channels, bias=True, dtype=DTYPE)

    def forward(self, hidden_states):
        residual = hidden_states
        b, c, h, w = hidden_states.shape
        n = F.group_norm(hidden_states, self.groups, eps=EPS)                      # affine folded into qkv
        x = n.view(b, c, h * w).transpose(1, 2)                                      # [B, HW, C] view
        q, k, v = F.linear(x, self.qkv.weight, self.qkv.bias).split(c, dim=-1)
        scores = torch.bmm(q, k.transpose(1, 2), out_dtype=torch.float32) * (c ** -0.5)
        probs = torch.softmax(scores, dim=-1).to(DTYPE)
        out = self.to_out(torch.bmm(probs, v))                                       # [B, HW, C]
        return out.transpose(1, 2).reshape(b, c, h, w) + residual


def convert_weights(sd):
    gamma, beta = sd["group_norm.weight"].float(), sd["group_norm.bias"].float()
    ws = [sd[f"to_{p}.weight"].float() for p in "qkv"]
    bs = [sd[f"to_{p}.bias"].float() for p in "qkv"]
    return {
        "qkv.weight": torch.cat([wi * gamma[None, :] for wi in ws], 0).to(DTYPE),
        "qkv.bias": torch.cat([wi @ beta + bi for wi, bi in zip(ws, bs)], 0).to(DTYPE),
        "to_out.weight": sd["to_out.weight"],
        "to_out.bias": sd["to_out.bias"],
    }
