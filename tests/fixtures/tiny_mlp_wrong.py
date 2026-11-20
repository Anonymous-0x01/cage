"""Illegal rewrite of tiny_mlp: GELU instead of SiLU (semantics change)."""

import torch.nn as nn
import torch.nn.functional as F
import torch

DTYPE = torch.float16


class Model(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False, dtype=DTYPE)
        self.up_proj = nn.Linear(hidden, inter, bias=False, dtype=DTYPE)
        self.down_proj = nn.Linear(inter, hidden, bias=False, dtype=DTYPE)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))
