"""RMSNorm (LLaMA-style) with residual add, FP16 activations, hidden=4096, batch 8 x seq 1024."""

import torch
import torch.nn as nn

BATCH, SEQ, HIDDEN = 8, 1024, 4096
DTYPE = torch.float16


class Model(nn.Module):
    def __init__(self, hidden: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden, dtype=DTYPE) + 0.01 * torch.randn(hidden, dtype=DTYPE))

    def forward(self, x: torch.Tensor, residual: torch.Tensor):
        h = x + residual
        hf = h.to(torch.float32)
        variance = hf.pow(2).mean(-1, keepdim=True)
        normed = hf * torch.rsqrt(variance + self.eps)
        return self.weight * normed.to(DTYPE), h


def get_init_inputs():
    return [HIDDEN]


def get_inputs():
    x = torch.randn(BATCH, SEQ, HIDDEN, dtype=DTYPE, device="cuda")
    residual = torch.randn(BATCH, SEQ, HIDDEN, dtype=DTYPE, device="cuda")
    return [x, residual]
INDUCTOR_CONFIG = {"coordinate_descent_tuning": True}
COMPILE_MODE = "max-autotune-no-cudagraphs"
