"""DeepSeek-V3 MoE router, G*: split-precision router GEMM + fusible group selection.

x is fp16 (exact in any wider format).  The fp32 router weight is split offline into
W_hi = fp16(W) and W_lo = fp16((W - W_hi) * 4096) (scaled to stay out of fp16 subnormals), so
W ~= W_hi + W_lo / 4096 with ~22 mantissa bits.  One fp16 tensor-core GEMM against [W_hi; W_lo]
with fp32 output and fp32 accumulation replaces the fp32 SIMT GEMM.
"""

import torch
import torch.nn as nn

N_EXPERTS, N_GROUP, TOPK_GROUP, TOP_K = 256, 8, 4, 8
ROUTED_SCALE = 2.5
LO_SCALE = 4096.0


class Model(nn.Module):
    def __init__(self, hidden: int, n_experts: int):
        super().__init__()
        self.w_split = nn.Parameter(torch.empty(2 * n_experts, hidden, dtype=torch.float16))
        self.e_score_correction_bias = nn.Parameter(torch.empty(n_experts, dtype=torch.float32))

    def forward(self, hidden_states):
        both = torch.mm(hidden_states, self.w_split.t(), out_dtype=torch.float32)
        logits = both[:, :N_EXPERTS] + both[:, N_EXPERTS:] * (1.0 / LO_SCALE)
        scores = logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        g = scores_for_choice.view(-1, N_GROUP, N_EXPERTS // N_GROUP)
        top1, arg1 = g.max(dim=-1, keepdim=True)
        top2 = g.scatter(-1, arg1, -float("inf")).amax(dim=-1, keepdim=True)
        group_scores = (top1 + top2).squeeze(-1)                       # == topk(2).sum(), same add order
        # group j is selected iff fewer than TOPK_GROUP groups beat it (ties broken by index)
        gi, gj = group_scores.unsqueeze(-1), group_scores.unsqueeze(-2)
        order = torch.arange(N_GROUP, device=g.device)
        beats = (gj > gi) | ((gj == gi) & (order.view(1, 1, -1) < order.view(1, -1, 1)))
        group_mask = beats.sum(-1) < TOPK_GROUP                          # [T, 8]
        score_mask = group_mask.unsqueeze(-1).expand(-1, N_GROUP, N_EXPERTS // N_GROUP).reshape(-1, N_EXPERTS)
        tmp_scores = scores_for_choice.masked_fill(~score_mask, 0.0)
        idx = torch.topk(tmp_scores, k=TOP_K, dim=-1, sorted=False)[1]
        # ascending order of the 8 selected ids via pairwise ranks
        pos = (idx.unsqueeze(-2) < idx.unsqueeze(-1)).sum(-1)            # rank of each id
        topk_idx = torch.zeros_like(idx).scatter(1, pos, idx)
        topk_weight = scores.gather(1, topk_idx)
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_idx, topk_weight * ROUTED_SCALE


def convert_weights(sd):
    w = sd["weight"].float()
    hi = w.half()
    lo = ((w - hi.float()) * LO_SCALE).half()
    return {"w_split": torch.cat([hi, lo], 0), "e_score_correction_bias": sd["e_score_correction_bias"]}
