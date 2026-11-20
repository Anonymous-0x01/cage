"""Swin shifted-window block, G*: offline attention bias + fused window attention + merged QKV.

* The relative-position bias (table gathered with a static index) and the cyclic-shift mask
  (built from arange each call) depend only on weights and config, so their sum is materialized once
  offline as attn_bias [num_windows, heads, 49, 49] (the original Swin implementation also registers the
  shift mask as a buffer).  The per-call gather / mask construction disappears from the graph.
* Window attention runs as one fused scaled_dot_product_attention call with that bias, batched as
  [B, num_windows * heads, 49, 32]; q/k/v come from one GEMM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

RES, WINDOW, SHIFT, MLP_RATIO, EPS = 56, 7, 3, 4.0, 1e-5
DTYPE = torch.float16


class Model(nn.Module):
    def __init__(self, dim: int, resolution: int, num_heads: int, window_size: int, shift_size: int):
        super().__init__()
        self.dim, self.num_heads, self.ws, self.shift, self.res = dim, num_heads, window_size, shift_size, resolution
        self.head_dim = dim // num_heads
        self.n_windows = (resolution // window_size) ** 2
        self.layernorm_before = nn.LayerNorm(dim, eps=EPS, dtype=DTYPE)
        self.layernorm_after = nn.LayerNorm(dim, eps=EPS, dtype=DTYPE)
        self.qkv = nn.Linear(dim, 3 * dim, bias=True, dtype=DTYPE)
        self.o_proj = nn.Linear(dim, dim, dtype=DTYPE)
        self.attn_bias = nn.Parameter(torch.empty(self.n_windows, num_heads, window_size ** 2, window_size ** 2,
                                                  dtype=DTYPE))
        self.fc1 = nn.Linear(dim, int(MLP_RATIO * dim), dtype=DTYPE)
        self.fc2 = nn.Linear(int(MLP_RATIO * dim), dim, dtype=DTYPE)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        b, _, c = hidden_states.shape
        ws, nh, hd, r, g = self.ws, self.num_heads, self.head_dim, self.res, self.res // self.ws
        L = ws * ws
        x = self.layernorm_before(hidden_states).view(b, r, r, c)
        x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
        x = x.view(b, g, ws, g, ws, c).transpose(2, 3).reshape(b * g * g, L, c)
        qkv = F.linear(x, self.qkv.weight, self.qkv.bias).view(b, g * g, L, 3, nh, hd)
        q, k, v = qkv.permute(3, 0, 1, 4, 2, 5).reshape(3, b, g * g * nh, L, hd).unbind(0)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=self.attn_bias.view(1, g * g * nh, L, L),
                                              scale=hd ** -0.5)
        attn = attn.reshape(b * g * g, nh, L, hd).transpose(1, 2).reshape(b * g * g, L, c)
        out = self.o_proj(attn).view(b, g, g, ws, ws, c).transpose(2, 3).reshape(b, r, r, c)
        out = torch.roll(out, shifts=(self.shift, self.shift), dims=(1, 2)).reshape(b, r * r, c)
        hidden_states = hidden_states + out
        return hidden_states + self.fc2(F.gelu(self.fc1(self.layernorm_after(hidden_states))))


def _window_partition(x, ws):
    b, h, w, c = x.shape
    return x.view(b, h // ws, ws, w // ws, ws, c).transpose(2, 3).contiguous().view(-1, ws, ws, c)


def convert_weights(sd):
    """Fold the relative-position bias gather and the shift mask into one offline bias (G0 computes both in fp16)."""
    sd = dict(sd)
    table = sd.pop("attention.relative_position_bias.relative_position_bias_table")      # [(2ws-1)^2, H]
    ws, shift, res = WINDOW, SHIFT, RES
    coords = torch.stack(torch.meshgrid([torch.arange(ws), torch.arange(ws)], indexing="ij")).flatten(1)
    rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
    rel[:, :, 0] += ws - 1
    rel[:, :, 1] += ws - 1
    rel[:, :, 0] *= 2 * ws - 1
    index = rel.sum(-1).view(-1).to(table.device)
    rel_bias = table[index].view(ws * ws, ws * ws, -1).permute(2, 0, 1).contiguous().unsqueeze(0)  # [1, H, L, L]

    idx = torch.arange(res, device=table.device)
    region = (idx >= res - ws).long() + (idx >= res - shift).long()
    img_mask = (region[None, :, None, None] * 3 + region[None, None, :, None]).to(table.dtype)
    mw = _window_partition(img_mask, ws).view(-1, ws * ws)
    mask = mw.unsqueeze(1) - mw.unsqueeze(2)
    mask = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)                  # [nW, L, L]
    sd["attn_bias"] = (rel_bias + mask.unsqueeze(1)).contiguous()                             # [nW, H, L, L]

    sd["qkv.weight"] = torch.cat([sd.pop(f"attention.{p}_proj.weight") for p in "qkv"], 0)
    sd["qkv.bias"] = torch.cat([sd.pop(f"attention.{p}_proj.bias") for p in "qkv"], 0)
    sd["o_proj.weight"] = sd.pop("attention.o_proj.weight")
    sd["o_proj.bias"] = sd.pop("attention.o_proj.bias")
    return sd
