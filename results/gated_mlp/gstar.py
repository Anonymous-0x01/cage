"""Gated MLP (SwiGLU) of a LLaMA-3-8B decoder layer with pre-RMSNorm and residual, FP16, 1 x 1024 tokens."""

import torch
import torch.nn as nn
import torch.nn.functional as F

TOKENS, HIDDEN, INTER = 1024, 4096, 14336
DTYPE = torch.float16


class Model(nn.Module):
    def __init__(self, hidden: int, inter: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.norm_weight = nn.Parameter(1 + 0.01 * torch.randn(hidden, dtype=DTYPE))
        self.gate_proj = nn.Linear(hidden, inter, bias=False, dtype=DTYPE)
        self.up_proj = nn.Linear(hidden, inter, bias=False, dtype=DTYPE)
        self.down_proj = nn.Linear(inter, hidden, bias=False, dtype=DTYPE)
        with torch.no_grad():
            for lin in (self.gate_proj, self.up_proj, self.down_proj):
                lin.weight.normal_(0, 0.02)

    def forward(self, x: torch.Tensor):
        residual = x
        hf = x.to(torch.float32)
        hf = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + self.eps)
        h = self.norm_weight * hf.to(DTYPE)
        gate = self.gate_proj(h)
        up = self.up_proj(h)
        act = F.silu(gate.float()).to(DTYPE) * up
        return residual + self.down_proj(act)


def get_init_inputs():
    return [HIDDEN, INTER]


def get_inputs():
    return [torch.randn(1, TOKENS, HIDDEN, dtype=DTYPE, device="cuda")]
COMPILE_MODE = "max-autotune-no-cudagraphs"
