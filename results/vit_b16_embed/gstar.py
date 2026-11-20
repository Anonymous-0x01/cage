"""ViT-B/16 embedding, G*: patchify convolution rewritten as reshape + GEMM.

A Conv2d with kernel == stride == 16 has non-overlapping receptive fields, so it equals
extracting 16x16 patches (a reshape/permute) and multiplying by the flattened kernel
[768, 3*16*16].  The GEMM output is already in [B, 196, 768] token layout, so the
flatten/transpose layout conversion disappears and bias, cls token, position embedding and
LayerNorm fuse into one pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPE = torch.float16


class Model(nn.Module):
    def __init__(self, dim: int, patch: int):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch, dtype=DTYPE)
        self.cls_token = nn.Parameter(torch.empty(1, 1, dim, dtype=DTYPE))
        self.pos_embed = nn.Parameter(torch.empty(1, 197, dim, dtype=DTYPE))
        self.norm = nn.LayerNorm(dim, eps=1e-6, dtype=DTYPE)

    def forward(self, x):
        b, c, hh, ww = x.shape
        p = self.patch
        gh, gw = hh // p, ww // p
        patches = x.view(b, c, gh, p, gw, p).permute(0, 2, 4, 1, 3, 5).reshape(b * gh * gw, c * p * p)
        w = self.proj.weight.view(self.proj.weight.shape[0], -1)
        tokens = F.linear(patches, w, self.proj.bias).view(b, gh * gw, -1)
        tokens = torch.cat([self.cls_token.expand(b, -1, -1), tokens], dim=1)
        return self.norm(tokens + self.pos_embed)
COMPILE_MODE = "max-autotune-no-cudagraphs"
