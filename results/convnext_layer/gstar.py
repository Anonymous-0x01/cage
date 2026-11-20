"""ConvNeXt block (ConvNeXt-T, stage 1), G*: layer-scale folded into pwconv2 + GEMM templates with GELU epilogue.

G0 below with two changes: gamma * (h @ W2^T + b2) == h @ (gamma W2)^T + gamma b2 is folded offline, and
COMPILE_MODE max-autotune lets Inductor fuse GELU into the pwconv1 GEMM epilogue.


Source: transformers 5.23.0, models/convnext/modeling_convnext.py, class ConvNextLayer (verbatim forward;
drop_path is identity at inference) with ConvNextLayerNorm (channels_last).
Config facebook/convnext-tiny-224: hidden_sizes[0] 96, hidden_act gelu, layer_scale_init_value 1e-6
(trained layer-scale values are larger, so they are drawn around 0.1 here).
Input: batch 64, 96 x 56 x 56, fp16.
"""

import torch
import torch.nn as nn

BATCH, DIM, RES = 64, 96, 56
DTYPE = torch.float16


class Model(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, dtype=DTYPE)  # depthwise conv
        self.layernorm = nn.LayerNorm(dim, eps=1e-6, dtype=DTYPE)
        self.pwconv1 = nn.Linear(dim, 4 * dim, dtype=DTYPE)
        self.pwconv2 = nn.Linear(4 * dim, dim, dtype=DTYPE)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        features = self.dwconv(features)
        features = features.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        features = self.layernorm(features)
        features = self.pwconv1(features)
        features = nn.functional.gelu(features)
        features = self.pwconv2(features)
        features = features.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        features = residual + features
        return features


def get_init_inputs():
    return [DIM]


def get_inputs():
    return [torch.randn(BATCH, DIM, RES, RES, dtype=DTYPE, device="cuda")]


COMPILE_MODE = "max-autotune-no-cudagraphs"


def convert_weights(sd):
    sd = dict(sd)
    gamma = sd.pop("layer_scale_parameter").float()
    sd["pwconv2.weight"] = (gamma[:, None] * sd["pwconv2.weight"].float()).to(DTYPE)
    sd["pwconv2.bias"] = (gamma * sd["pwconv2.bias"].float()).to(DTYPE)
    return sd
