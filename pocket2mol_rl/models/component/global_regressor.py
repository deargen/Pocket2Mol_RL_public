import torch
import torch.nn as nn
from torch.nn import Module, Sequential

from pocket2mol_rl.utils.tensor import decompose_tensors

from .invariant import GVLinear, GVPerceptronVN


class GlobalRegressorVN(Module):
    def __init__(self, in_sca, in_vec, hidden_dim_sca, hidden_dim_vec):
        super().__init__()
        self.net = Sequential(
            GVPerceptronVN(in_sca, in_vec, hidden_dim_sca, hidden_dim_vec),
            GVLinear(hidden_dim_sca, hidden_dim_vec, hidden_dim_sca, None),
        )

        self.mlp_before_pooling = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim_sca, 4 * hidden_dim_sca),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.mlp_after_pooling = nn.Sequential(
            nn.Linear(4 * hidden_dim_sca, hidden_dim_sca),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim_sca, 1),
        )

    def forward(self, h_att, partition=None):
        x = self.net(h_att)[0]
        x = self.mlp_before_pooling(x)
        if partition is None:
            x = x.mean(dim=0)  # mean pooling
            return self.mlp_after_pooling(x)
        else:
            xs = decompose_tensors(x, partition, dim=0)
            x = torch.stack([x.mean(dim=0) for x in xs], dim=0)
            return self.mlp_after_pooling(x)
