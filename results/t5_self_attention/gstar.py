"""T5 encoder self-attention, G*: relative position bias as an offline distance table inside flex_attention.

The eager path buckets an [S, S] relative-position grid (log / min / where on int64), gathers an
[S, S, H] bias, permutes it to [1, H, S, S], adds it to materialized [B, H, S, S] scores and runs a
bf16 softmax.  The bias only depends on d = kv_idx - q_idx, and T5's bucket function saturates for
|d| >= 91 (bucket 15 / 31 up to max_distance 128 and beyond), so the whole bias is a [H, 257] table over
d in [-128, 128] built offline from the embedding weights; one fused flex_attention kernel adds
table[h, clamp(d, -128, 128) + 128] to each score.  q/k/v share one GEMM.
"""

import math

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention

N_BUCKETS, MAX_DIST, EPS, CLAMP = 32, 128, 1e-6, 128
DTYPE = torch.bfloat16


def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
    )
    relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets


class Model(nn.Module):
    def __init__(self, d_model: int, d_kv: int, n_heads: int):
        super().__init__()
        self.d_kv, self.n_heads = d_kv, n_heads
        self.layer_norm_weight = nn.Parameter(torch.empty(d_model, dtype=DTYPE))
        self.qkv = nn.Linear(d_model, 3 * n_heads * d_kv, bias=False, dtype=DTYPE)
        self.o = nn.Linear(n_heads * d_kv, d_model, bias=False, dtype=DTYPE)
        self.rel_table = nn.Parameter(torch.empty(n_heads, 2 * CLAMP + 1, dtype=torch.float32))

    def forward(self, hidden_states):
        b, s, _ = hidden_states.shape
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        normed = self.layer_norm_weight * (hidden_states * torch.rsqrt(variance + EPS)).to(DTYPE)
        q, k, v = self.qkv(normed).view(b, s, 3, self.n_heads, self.d_kv).permute(2, 0, 3, 1, 4).unbind(0)
        table = self.rel_table

        def rel_bias(score, bi, h, q_idx, kv_idx):
            return score + table[h, torch.clamp(kv_idx - q_idx, -CLAMP, CLAMP) + CLAMP]

        out = flex_attention(q, k, v, score_mod=rel_bias, scale=1.0)
        return hidden_states + self.o(out.transpose(1, 2).reshape(b, s, -1))


def convert_weights(sd):
    sd = dict(sd)
    weight = sd.pop("relative_attention_bias.weight")                               # [32 buckets, H] bf16
    distance = torch.arange(-CLAMP, CLAMP + 1, device=weight.device)
    bucket = _relative_position_bucket(distance, bidirectional=True, num_buckets=N_BUCKETS, max_distance=MAX_DIST)
    sd["rel_table"] = weight[bucket].t().float().contiguous()                        # exact bf16 values
    sd["qkv.weight"] = torch.cat([sd.pop("q.weight"), sd.pop("k.weight"), sd.pop("v.weight")], 0)
    return sd
