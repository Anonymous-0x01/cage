"""Qwen2-VL vision patch embedding, G*: the patch Conv3d is a GEMM.

Every row of pixel_values is exactly one (C, T, P, P) patch and the Conv3d has kernel == stride == patch,
so its output is a single 1x1x1 position: out = patch_vector @ W.view(embed_dim, C*T*P*P)^T.  The view
order (C, T, P, P) of the input matches the flattening order of the weight, so this is exact.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPE = torch.bfloat16


class Model(nn.Module):
    def __init__(self, patch_size: int, temporal_patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False,
                              dtype=DTYPE)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        w = self.proj.weight.view(self.embed_dim, -1)
        return F.linear(hidden_states.to(dtype=w.dtype), w)
