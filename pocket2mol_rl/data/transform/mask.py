import copy
import random

import numpy as np
import torch
from easydict import EasyDict
from torch_geometric.utils.subgraph import subgraph

from pocket2mol_rl.data.data import ProteinLigandData

from .featurize import FeaturizeLigandAtom


class LigandMaskBase:
    def __init__(
        self,
        min_ratio=0.0,
        max_ratio=1.2,
        min_num_masked=1,
        min_num_unmasked=0,
        featurize_module=FeaturizeLigandAtom(),
    ):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_num_masked = min_num_masked
        self.min_num_unmasked = min_num_unmasked
        self.featurize_module = featurize_module

    def handle_data(
        self, data: ProteinLigandData, masked_idx, context_idx
    ) -> ProteinLigandData:
        data.context_idx = context_idx
        data.masked_idx = masked_idx
        data.ligand_masked_element = data.ligand_element[masked_idx]
        data.ligand_masked_pos = data.ligand_pos[masked_idx]
        data.ligand_context_element = data.ligand_element[context_idx]
        data.ligand_context_feature_full = data.ligand_atom_feature_full[context_idx]
        data.ligand_context_pos = data.ligand_pos[context_idx]

        if data.ligand_bond_index.size(1) != 0:
            data.ligand_context_bond_index, data.ligand_context_bond_type = subgraph(
                context_idx,
                data.ligand_bond_index,
                edge_attr=data.ligand_bond_type,
                relabel_nodes=True,
            )
        else:
            data.ligand_context_bond_index = torch.empty([2, 0], dtype=torch.long)
            data.ligand_context_bond_type = torch.empty([0], dtype=torch.long)

        data = self.featurize_module.change_features_of_neigh(data)
        data.ligand_frontier = (
            data.ligand_context_num_neighbors
            < data.ligand_num_neighbors[data.context_idx]
        )

        return data

    def __call__(self, data: ProteinLigandData) -> ProteinLigandData:
        masked_idx, context_idx = self.mask_idx(data)
        data = self.handle_data(data, masked_idx, context_idx)
        return data

    def mask_idx(self, data: ProteinLigandData):
        """Masking ligand atoms. Random or BFS."""
        raise NotImplementedError


class LigandRandomMask(LigandMaskBase):
    def mask_idx(self, data):
        ratio = np.clip(random.uniform(self.min_ratio, self.max_ratio), 0.0, 1.0)
        num_atoms = data.ligand_element.size(0)
        num_masked = int(num_atoms * ratio)

        if num_masked < self.min_num_masked:
            num_masked = self.min_num_masked
        if (num_atoms - num_masked) < self.min_num_unmasked:
            num_masked = num_atoms - self.min_num_unmasked

        idx = np.arange(num_atoms)
        np.random.shuffle(idx)
        idx = torch.LongTensor(idx)
        masked_idx = idx[:num_masked]
        context_idx = idx[num_masked:]
        return masked_idx, context_idx


class LigandMaskNone(LigandMaskBase):
    def mask_idx(self, data):
        num_atoms = len(data.ligand_pos)

        idx = torch.LongTensor(np.arange(num_atoms))
        return idx[:0], idx


class LigandBFSMask(LigandMaskBase):
    def __init__(self, inverse=False, **kwargs):
        super().__init__(**kwargs)
        self.inverse = inverse

    @staticmethod
    def get_bfs_perm(nbh_list):
        num_nodes = len(nbh_list)
        num_neighbors = torch.LongTensor([len(nbh_list[i]) for i in range(num_nodes)])

        bfs_queue = [random.randint(0, num_nodes - 1)]
        bfs_perm = []
        num_remains = [num_neighbors.clone()]
        bfs_next_list = {}
        visited = {bfs_queue[0]}

        num_nbh_remain = num_neighbors.clone()

        while len(bfs_queue) > 0:
            current = bfs_queue.pop(0)
            for nbh in nbh_list[current]:
                num_nbh_remain[nbh] -= 1
            bfs_perm.append(current)
            num_remains.append(num_nbh_remain.clone())
            next_candid = []
            for nxt in nbh_list[current]:
                if nxt in visited:
                    continue
                next_candid.append(nxt)
                visited.add(nxt)

            random.shuffle(next_candid)
            bfs_queue += next_candid
            bfs_next_list[current] = copy.copy(bfs_queue)

        return torch.LongTensor(bfs_perm), bfs_next_list, num_remains

    def mask_idx(self, data):
        bfs_perm, bfs_next_list, num_remaining_nbs = self.get_bfs_perm(
            data.ligand_nbh_list
        )

        ratio = np.clip(random.uniform(self.min_ratio, self.max_ratio), 0.0, 1.0)
        num_atoms = data.ligand_element.size(0)
        num_masked = max(
            min(int(num_atoms * ratio), num_atoms - self.min_num_unmasked),
            self.min_num_masked,
        )
        if self.inverse:
            masked_idx, context_idx = bfs_perm[:num_masked], bfs_perm[num_masked:]
        else:
            masked_idx, context_idx = bfs_perm[-num_masked:], bfs_perm[:-num_masked]

        return masked_idx, context_idx


class LigandMaskAll(LigandRandomMask):
    def __init__(self, **kwargs):
        super().__init__(min_ratio=1.0, **kwargs)


class LigandMixedMask:
    def __init__(self, p_random=0.5, p_bfs=0.25, p_invbfs=0.25, **kwargs):
        super().__init__()

        self.t = [
            LigandRandomMask(**kwargs),
            LigandBFSMask(inverse=False, **kwargs),
            LigandBFSMask(inverse=True, **kwargs),
        ]
        self.p = [p_random, p_bfs, p_invbfs]

    def __call__(self, data):
        f = random.choices(self.t, k=1, weights=self.p)[0]
        return f(data)


def get_mask(cfg: EasyDict, **kwargs) -> LigandMaskBase:
    """Get mask class from config

    Todo:
        Pass config directly instead of `**mask_kwargs`

    Args:
        cfg (EasyDict): Config
        `**kwargs`: Additional arguments (e.g. featurize_module)

    Raises:
        NotImplementedError: If type is invalid

    Returns:
        LigandMaskBase: Initialized mask class
    """
    mask_class_dict = {
        "bfs": LigandBFSMask,
        "random": LigandRandomMask,
        "mixed": LigandMixedMask,
        "all": LigandMaskAll,
    }
    common_kwargs = {
        "min_ratio": cfg.min_ratio,
        "max_ratio": cfg.max_ratio,
        "min_num_masked": cfg.min_num_masked,
        "min_num_unmasked": cfg.min_num_unmasked,
    }
    # if type is invalid, raise error
    if not cfg.type in mask_class_dict:
        raise NotImplementedError("Unknown mask: %s" % cfg.type)

    # add kwargs for each mask type
    if cfg.type == "mixed":
        # add p_random, p_bfs, p_invbfs
        mask_kwargs = {
            **common_kwargs,
            "p_random": cfg.p_random,
            "p_bfs": cfg.p_bfs,
            "p_invbfs": cfg.p_invbfs,
        }
    elif cfg.type == "all":
        # remove all kwargs
        mask_kwargs = {}
    else:
        mask_kwargs = common_kwargs

    return mask_class_dict[cfg.type](**mask_kwargs, **kwargs)
