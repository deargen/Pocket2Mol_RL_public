import torch
from torch.nn import Module, Sequential

from .invariant import GVLinear, GVPerceptronVN


class FrontierLayerVN(Module):
    def __init__(self, in_sca, in_vec, hidden_dim_sca, hidden_dim_vec):
        super().__init__()
        self.net = Sequential(
            GVPerceptronVN(in_sca, in_vec, hidden_dim_sca, hidden_dim_vec),
            GVLinear(hidden_dim_sca, hidden_dim_vec, 1, 1),
        )

    def forward(self, h_att, idx_ligans):
        h_att_ligand = [h_att[0][idx_ligans], h_att[1][idx_ligans]]
        pred = self.net(h_att_ligand)
        pred = pred[0]
        return pred


class NoFrontierLayerVN(Module):
    def __init__(self, in_sca, in_vec, hidden_dim_sca, hidden_dim_vec):
        super().__init__()
        self.net = Sequential(
            GVPerceptronVN(in_sca, in_vec, hidden_dim_sca, hidden_dim_vec),
            GVLinear(hidden_dim_sca, hidden_dim_vec, 1, 1),
        )

    def load_params_from(self, model: FrontierLayerVN):
        if type(model) != FrontierLayerVN:
            raise NotImplementedError()

        self.net.load_state_dict(model.net.state_dict())

    def forward(self, h_att, idx_ligans):
        h_att_ligand = [h_att[0][idx_ligans], h_att[1][idx_ligans]]
        pred = self.net(h_att_ligand)
        pred = pred[0]

        assert pred.shape == (len(idx_ligans), 1)
        pred = -torch.mean(pred, dim=0)

        return pred
