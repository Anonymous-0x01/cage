"""DeepSeek-V3 MoE router (noaux_tc gating), 4096 tokens.

fp32 router GEMM (7168 -> 256 experts) -> sigmoid -> + e_score_correction_bias -> group-limited
selection (8 groups, keep the 4 groups with the best top-2 sums) -> top-8 experts -> gather
original sigmoid scores -> normalize -> x routed_scaling_factor 2.5.  Follows the reference
`DeepseekV3TopkRouter`/`MoEGate`; selected expert ids are returned in ascending order (the
consumer treats them as a set) so that equivalent implementations produce identical outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

TOKENS, HIDDEN, N_EXPERTS, N_GROUP, TOPK_GROUP, TOP_K = 4096, 7168, 256, 8, 4, 8
ROUTED_SCALE = 2.5
DTYPE = torch.float16


class Model(nn.Module):
    def __init__(self, hidden: int, n_experts: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_experts, hidden, dtype=torch.float32) * hidden ** -0.5)
        self.e_score_correction_bias = nn.Parameter(0.1 * torch.randn(n_experts, dtype=torch.float32))

    def forward(self, hidden_states):
        logits = F.linear(hidden_states.to(torch.float32), self.weight)
        scores = logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        group_scores = scores_for_choice.view(-1, N_GROUP, N_EXPERTS // N_GROUP).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(-1, N_GROUP, N_EXPERTS // N_GROUP).reshape(-1, N_EXPERTS)
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_idx = torch.topk(tmp_scores, k=TOP_K, dim=-1, sorted=False)[1]
        topk_idx = topk_idx.sort(dim=-1)[0]
        topk_weight = scores.gather(1, topk_idx)
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_idx, topk_weight * ROUTED_SCALE


def get_init_inputs():
    return [HIDDEN, N_EXPERTS]


def get_inputs():
    return [torch.randn(TOKENS, HIDDEN, dtype=DTYPE, device="cuda")]
