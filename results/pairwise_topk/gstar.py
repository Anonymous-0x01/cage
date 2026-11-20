"""PairwiseTopK candidate: G2 fp16 ranking (tensor-core bmm, 16-bit topk) + fp32 exact rerank of a shortlist."""

import torch
import torch.nn as nn

DTYPE = torch.float16
SHORTLIST = 32


class Model(nn.Module):
    def __init__(self, k: int, tau: float = 4.0):
        super().__init__()
        self.k, self.tau = k, tau

    def forward(self, q, keys, values):
        b, m, d = q.shape
        n = keys.shape[1]
        qf, kf = q.float(), keys.float()
        # ||q-k||^2 = ||q||^2 + ||k||^2 - 2 q.k ; ||q||^2 is constant per row, so it does not change the ranking
        # ranking only needs to be roughly right: the shortlist is re-ranked exactly below
        approx = ((kf * kf).sum(-1).unsqueeze(1) - 2.0 * torch.bmm(q, keys.transpose(1, 2)).float()).to(torch.float16)
        cand = approx.topk(SHORTLIST, dim=-1, largest=False, sorted=False).indices          # [B, M, S]
        # exact distances (G0's formula) only for the shortlist: robust to cancellation in the expansion
        kc = torch.gather(keys.unsqueeze(1).expand(b, m, n, d), 2, cand.unsqueeze(-1).expand(b, m, SHORTLIST, d))
        exact = (qf.unsqueeze(2) - kc.float()).pow(2).sum(-1)                                # [B, M, S]
        topd, j = exact.topk(self.k, dim=-1, largest=False, sorted=True)
        idx = torch.gather(cand, 2, j)
        w = torch.softmax(-topd / self.tau, dim=-1)
        gathered = torch.gather(values.unsqueeze(1).expand(b, m, -1, -1), 2,
                                idx.unsqueeze(-1).expand(b, m, self.k, values.shape[-1]))
        out = (w.unsqueeze(-1) * gathered.float()).sum(2)
        return topd, out.to(DTYPE)
COMPILE_MODE = "reduce-overhead"
