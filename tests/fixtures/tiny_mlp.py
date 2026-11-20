"""Tiny gated MLP task (G0) for harness tests: separate gate/up projections."""

import torch
import torch.nn as nn
import torch.nn.functional as F

TOKENS, HIDDEN, INTER = 64, 256, 512
DTYPE = torch.float16


class Model(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False, dtype=DTYPE)
        self.up_proj = nn.Linear(hidden, inter, bias=False, dtype=DTYPE)
        self.down_proj = nn.Linear(inter, hidden, bias=False, dtype=DTYPE)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def get_init_inputs():
    return [HIDDEN, INTER]


def get_inputs():
    return [torch.randn(TOKENS, HIDDEN, dtype=DTYPE, device="cuda")]
