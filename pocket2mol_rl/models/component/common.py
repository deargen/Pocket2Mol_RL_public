import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


def concat_tensors_to_batch(x_split):
    x = torch.cat(x_split, dim=0)
    batch = torch.repeat_interleave(
        torch.arange(len(x_split)),
        repeats=torch.LongTensor([s.size(0) for s in x_split]),
    ).to(device=x.device)
    return x, batch


def split_tensor_to_segments(x, segsize):
    num_segs = math.ceil(x.size(0) / segsize)
    segs = []
    for i in range(num_segs):
        segs.append(x[i * segsize : (i + 1) * segsize])
    return segs


def split_tensor_by_lengths(x, lengths):
    segs = []
    for l in lengths:
        segs.append(x[:l])
        x = x[l:]
    return segs


def batch_intersection_mask(batch, batch_filter):
    batch_filter = batch_filter.unique()
    mask = (batch.view(-1, 1) == batch_filter.view(1, -1)).any(dim=1)
    return mask


def get_batch_edge(ligand_context_bond_index, ligand_context_bond_type):
    return ligand_context_bond_index, ligand_context_bond_type


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction="mean", smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            if n_classes == 1:
                _smoothing = smoothing
            else:
                _smoothing = smoothing / (n_classes - 1)
            targets = (
                torch.empty(size=(targets.size(0), n_classes), device=targets.device)
                .fill_(_smoothing)
                .scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
            )
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(
            targets, inputs.size(-1), self.smoothing
        )
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss


class EdgeExpansion(nn.Module):
    def __init__(self, edge_channels):
        super().__init__()
        self.nn = nn.Linear(in_features=1, out_features=edge_channels, bias=False)

    def forward(self, edge_vector):
        edge_vector = edge_vector / (
            torch.norm(edge_vector, p=2, dim=1, keepdim=True) + 1e-7
        )
        expansion = self.nn(edge_vector.unsqueeze(-1)).transpose(1, -1)
        return expansion


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super().__init__()
        self.stop = stop
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.clamp_max(self.stop)
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


def embed_compose(
    compose_feature,
    compose_pos,
    idx_ligand,
    idx_protein,
    ligand_atom_emb,
    protein_atom_emb,
    emb_dim,
    space_dim=3,
):
    h_ligand = ligand_atom_emb(compose_feature[idx_ligand], compose_pos[idx_ligand])
    h_protein = protein_atom_emb(compose_feature[idx_protein], compose_pos[idx_protein])

    h_sca = torch.zeros(
        [len(compose_pos), emb_dim[0]],
    ).to(h_ligand[0])
    h_vec = torch.zeros(
        [len(compose_pos), emb_dim[1], space_dim],
    ).to(h_ligand[1])
    h_sca[idx_ligand], h_sca[idx_protein] = h_ligand[0], h_protein[0]
    h_vec[idx_ligand], h_vec[idx_protein] = h_ligand[1], h_protein[1]
    return [h_sca, h_vec]
