"""ViT-B/16 embedding + first pre-norm (timm `PatchEmbed` + cls token + pos embed + LayerNorm), batch 64.

Conv2d 16x16/16 patchify (3 -> 768) -> flatten(2).transpose(1, 2) -> prepend cls token -> add
position embedding -> LayerNorm.  FP16, 224x224 images.  The CAGE paper's motivation lists ViT
patch embedding as a case where TorchInductor is slower than eager because of layout conversions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH, IMG, PATCH, DIM = 64, 224, 16, 768
N_PATCH = (IMG // PATCH) ** 2
DTYPE = torch.float16


class Model(nn.Module):
    def __init__(self, dim: int, patch: int):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch, dtype=DTYPE)
        self.cls_token = nn.Parameter(0.02 * torch.randn(1, 1, dim, dtype=DTYPE))
        self.pos_embed = nn.Parameter(0.02 * torch.randn(1, N_PATCH + 1, dim, dtype=DTYPE))
        self.norm = nn.LayerNorm(dim, eps=1e-6, dtype=DTYPE)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = x + self.pos_embed
        return self.norm(x)


def get_init_inputs():
    return [DIM, PATCH]


def get_inputs():
    return [torch.randn(BATCH, 3, IMG, IMG, dtype=DTYPE, device="cuda")]
