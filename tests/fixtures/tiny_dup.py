"""Tiny task whose G0 recomputes the same projection twice (duplicated sub-expression)."""

import torch
import torch.nn as nn

DTYPE = torch.float16


class Model(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=False, dtype=DTYPE)

    def forward(self, x):
        a = self.proj(x)
        b = self.proj(x)
        return a + b * 0.5


def get_init_inputs():
    return [1024]


def get_inputs():
    return [torch.randn(2048, 1024, dtype=DTYPE, device="cuda")]
