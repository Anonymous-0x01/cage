"""EdgeProjAgg candidate: G1.3 one node GEMM for source+destination projections."""

import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPE = torch.float16


class Model(nn.Module):
    def __init__(self, n_nodes: int, hidden: int, edge_dim: int):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden = hidden
        self.w_node = nn.Parameter(torch.empty(2 * hidden, hidden, dtype=DTYPE))
        self.w_edge = nn.Parameter(torch.empty(hidden, edge_dim, dtype=DTYPE))
        self.msg_bias = nn.Parameter(torch.empty(hidden, dtype=DTYPE))
        self.out_proj = nn.Linear(hidden, hidden, dtype=DTYPE)

    def forward(self, x, src, dst, edge_attr):
        # (x @ W)[idx] == x[idx] @ W: project the 16x fewer nodes, then gather
        p_src, p_dst = F.linear(x, self.w_node).split(self.hidden, dim=-1)
        pre = p_src[src] + p_dst[dst] + F.linear(edge_attr, self.w_edge, self.msg_bias)
        msg = F.gelu(pre)
        agg = torch.zeros(self.n_nodes, msg.shape[-1], dtype=torch.float32, device=x.device)
        agg = agg.index_add(0, dst, msg.float())
        deg = torch.zeros(self.n_nodes, dtype=torch.float32, device=x.device)
        deg = deg.index_add(0, dst, torch.ones_like(dst, dtype=torch.float32))
        agg = agg / deg.clamp(min=1.0).unsqueeze(-1)
        return x + self.out_proj(agg.to(DTYPE))


def convert_weights(sd):
    w = sd["msg_proj.weight"]
    h = w.shape[0]
    return {
        "w_node": torch.cat([w[:, :h], w[:, h:2 * h]], dim=0),
        "w_edge": w[:, 2 * h:].contiguous(),
        "msg_bias": sd["msg_proj.bias"],
        "out_proj.weight": sd["out_proj.weight"],
        "out_proj.bias": sd["out_proj.bias"],
    }
COMPILE_MODE = "max-autotune-no-cudagraphs"
