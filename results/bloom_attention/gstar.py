"""BLOOM attention, G*: ALiBi + causal attention as one flex_attention call.

The eager path builds alibi [B*H, 1, S] and a dense [S, S] causal mask, materializes
baddbmm(alibi, q, k^T) scores [B, H, S, S] in bf16, adds the mask and runs an fp32 softmax.
ALiBi is a per-head linear bias on the key position, i.e. score_mod(score) = score + slope_h * kv_idx,
and the causal mask is a block mask: one fused attention kernel, masked blocks skipped, scores kept
in fp32 (the bf16 rounding of alibi + q.k in G0 is the dominant error source at long positions).
"""

import math

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

EPS = 1e-5
DTYPE = torch.bfloat16


def _get_alibi_slopes(num_heads: int, device: torch.device) -> torch.Tensor:
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.full((), 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=device, dtype=torch.float32)
    powers = torch.arange(1, 1 + closest_power_of_2, device=device, dtype=torch.int32)
    slopes = torch.pow(base, powers)
    if closest_power_of_2 != num_heads:
        extra_base = torch.full(
            (), 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


class Model(nn.Module):
    def __init__(self, hidden_size: int, n_head: int):
        super().__init__()
        self.num_heads = n_head
        self.head_dim = hidden_size // n_head
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=EPS, dtype=DTYPE)
        self.query_key_value = nn.Linear(hidden_size, 3 * hidden_size, bias=True, dtype=DTYPE)
        self.dense = nn.Linear(hidden_size, hidden_size, dtype=DTYPE)

    def forward(self, hidden_states):
        b, s, _ = hidden_states.shape
        slopes = _get_alibi_slopes(self.num_heads, hidden_states.device)

        def alibi(score, bi, h, q_idx, kv_idx):
            return score + slopes[h] * kv_idx

        fused_qkv = self.query_key_value(self.input_layernorm(hidden_states))
        qkv = fused_qkv.view(b, s, self.num_heads, 3, self.head_dim)
        q, k, v = (qkv[..., i, :].transpose(1, 2) for i in range(3))
        block_mask = create_block_mask(causal, None, None, s, s, device=hidden_states.device)
        ctx = flex_attention(q, k, v, score_mod=alibi, block_mask=block_mask, scale=self.inv_norm_factor)
        return hidden_states + self.dense(ctx.transpose(1, 2).reshape(b, s, -1))
