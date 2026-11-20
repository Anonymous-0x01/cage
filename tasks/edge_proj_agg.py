"""EdgeProjectionAggregation: message passing over a sparse graph (GNN / graph-transformer layer).

Each edge builds its message from the concatenated source features, destination features and
edge attributes, projects it, applies GELU, and the messages are mean-aggregated into the
destination nodes, followed by an output projection with a residual connection.
FP16, 16384 nodes, 262144 edges (average in-degree 16), hidden 256, edge features 64.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_NODES, N_EDGES, HIDDEN, EDGE_DIM = 16384, 262144, 256, 64
DTYPE = torch.float16


class Model(nn.Module):
    def __init__(self, n_nodes: int, hidden: int, edge_dim: int):
        super().__init__()
        self.n_nodes = n_nodes
        self.msg_proj = nn.Linear(2 * hidden + edge_dim, hidden, dtype=DTYPE)
        self.out_proj = nn.Linear(hidden, hidden, dtype=DTYPE)

    def forward(self, x, src, dst, edge_attr):
        h = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        msg = F.gelu(self.msg_proj(h))
        agg = torch.zeros(self.n_nodes, msg.shape[-1], dtype=torch.float32, device=x.device)
        agg = agg.index_add(0, dst, msg.float())
        deg = torch.zeros(self.n_nodes, dtype=torch.float32, device=x.device)
        deg = deg.index_add(0, dst, torch.ones_like(dst, dtype=torch.float32))
        agg = agg / deg.clamp(min=1.0).unsqueeze(-1)
        return x + self.out_proj(agg.to(DTYPE))


def get_init_inputs():
    return [N_NODES, HIDDEN, EDGE_DIM]


def get_inputs():
    x = torch.randn(N_NODES, HIDDEN, dtype=DTYPE, device="cuda")
    src = torch.randint(0, N_NODES, (N_EDGES,), device="cuda")
    dst = torch.randint(0, N_NODES, (N_EDGES,), device="cuda")
    edge_attr = torch.randn(N_EDGES, EDGE_DIM, dtype=DTYPE, device="cuda")
    return [x, src, dst, edge_attr]
