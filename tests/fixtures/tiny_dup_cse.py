"""Rewrite of tiny_dup: the projection is computed once (a + 0.5 * a == 1.5 * a)."""

import torch
import torch.nn as nn

DTYPE = torch.float16


class Model(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=False, dtype=DTYPE)

    def forward(self, x):
        return self.proj(x) * 1.5
