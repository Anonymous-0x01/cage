"""Qwen2-VL vision patch embedding (Qwen2-VL-7B-Instruct vision tower).

Source: transformers 5.23.0, models/qwen2_vl/modeling_qwen2_vl.py, class PatchEmbed (verbatim forward).
Config (Qwen2-VL-7B vision_config): patch_size 14, temporal_patch_size 2, in_chans 3, embed_dim 1280,
bf16 weights.  The processor delivers pixel_values as fp32 [num_patches, 3*2*14*14]; 16384 patches
(e.g. four 896x896 images: 4 x 64 x 64).
"""

import torch
import torch.nn as nn

N_PATCHES, PATCH, TEMPORAL, IN_CH, EMBED = 16384, 14, 2, 3, 1280
DTYPE = torch.bfloat16


class Model(nn.Module):
    def __init__(self, patch_size: int, temporal_patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False,
                              dtype=DTYPE)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


def get_init_inputs():
    return [PATCH, TEMPORAL, IN_CH, EMBED]


def get_inputs():
    return [torch.randn(N_PATCHES, IN_CH * TEMPORAL * PATCH * PATCH, dtype=torch.float32, device="cuda")]
