"""PairwiseTopK: k-nearest-neighbour retrieval with distance-weighted aggregation.

For every query, compute squared Euclidean distances to all keys, select the K nearest,
and aggregate their values with softmax(-distance / tau) weights (point-cloud / retrieval
attention style).  Returns the sorted top-K distances and the aggregated values.
Queries/keys/values FP16, distances in FP32; batch 2, 1024 queries, 4096 keys, dim 32, K=16.
"""

import torch
import torch.nn as nn

BATCH, N_QUERY, N_KEY, DIM, VDIM, TOPK = 2, 1024, 4096, 32, 64, 16
DTYPE = torch.float16


class Model(nn.Module):
    def __init__(self, k: int, tau: float = 4.0):
        super().__init__()
        self.k, self.tau = k, tau

    def forward(self, q, keys, values):
        diff = q.float().unsqueeze(2) - keys.float().unsqueeze(1)          # [B, M, N, D]
        dist = diff.pow(2).sum(-1)                                          # [B, M, N]
        topd, idx = dist.topk(self.k, dim=-1, largest=False, sorted=True)   # [B, M, K]
        w = torch.softmax(-topd / self.tau, dim=-1)
        b, m, k = idx.shape
        gathered = torch.gather(values.unsqueeze(1).expand(b, m, -1, -1), 2,
                                idx.unsqueeze(-1).expand(b, m, k, values.shape[-1]))  # [B, M, K, Dv]
        out = (w.unsqueeze(-1) * gathered.float()).sum(2)
        return topd, out.to(DTYPE)


def get_init_inputs():
    return [TOPK]


def get_inputs():
    q = torch.randn(BATCH, N_QUERY, DIM, dtype=DTYPE, device="cuda")
    keys = torch.randn(BATCH, N_KEY, DIM, dtype=DTYPE, device="cuda")
    values = torch.randn(BATCH, N_KEY, VDIM, dtype=DTYPE, device="cuda")
    return [q, keys, values]
