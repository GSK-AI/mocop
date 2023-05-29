import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_edge: int = 1, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_edge, in_dim, out_dim))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_edge, 1, out_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, adj: torch.Tensor, feat: torch.Tensor):
        if len(adj.size()) == 3:
            adj = adj.unsqueeze(1)
        feat = feat.unsqueeze(1)
        output = torch.matmul(adj, feat)
        output = torch.matmul(output, self.weight)
        if self.bias is not None:
            output = output + self.bias
        output = output.sum(dim=1)
        output = F.relu(output)
        return output


class GatedGraphConvolution(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_edge: int = 1, bias: bool = True):
        super(GatedGraphConvolution, self).__init__()
        if in_dim != out_dim:
            raise ValueError(
                in_dim == out_dim,
                f"Input ({in_dim}) and output "
                "({out_dim}) must have the same dimension.",
            )
        self.gc = GraphConvolution(
            in_dim=in_dim, out_dim=out_dim, n_edge=n_edge, bias=bias
        )
        self.gru = GRU2D(in_dim=out_dim, hidden_dim=out_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.gc.reset_parameters()
        self.gru.reset_parameters()

    def forward(self, adj: torch.Tensor, h_0: torch.Tensor):
        h = self.gc(adj, h_0)
        output = self.gru(h, h_0)
        return output


class GRU2D(nn.Module):
    """2D GRU Cell"""

    def __init__(self, in_dim, hidden_dim, bias=True):
        super(GRU2D, self).__init__()
        self.x_to_intermediate = nn.Linear(in_dim, 3 * hidden_dim, bias=bias)
        self.h_to_intermediate = nn.Linear(in_dim, 3 * hidden_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        for k, v in self.state_dict().items():
            if "weight" in k:
                std = math.sqrt(6.0 / (v.size(1) + v.size(0) / 3))
                nn.init.uniform_(v, a=-std, b=std)
            elif "bias" in k:
                nn.init.zeros_(v)

    def forward(self, x: torch.Tensor, h_0: torch.Tensor):
        intermediate_x = self.x_to_intermediate(x)
        intermediate_h = self.h_to_intermediate(h_0)

        x_r, x_z, x_n = intermediate_x.chunk(3, -1)
        h_r, h_z, h_n = intermediate_h.chunk(3, -1)

        r = torch.sigmoid(x_r + h_r)
        z = torch.sigmoid(x_z + h_z)
        n = torch.tanh(x_n + (r * h_n))
        h = (1 - z) * n + z * h_0

        return h
