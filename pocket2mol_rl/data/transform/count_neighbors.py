import torch
from torch_geometric.utils.num_nodes import maybe_num_nodes
from pocket2mol_rl.utils.deterministic import scatter_add


class LigandCountNeighbors(object):
    @staticmethod
    def count_neighbors(edge_index, symmetry, valence=None, num_nodes=None):
        assert symmetry == True, "Only support symmetrical edges."

        if num_nodes is None:
            num_nodes = maybe_num_nodes(edge_index)

        if valence is None:
            valence = torch.ones([edge_index.size(1)], device=edge_index.device)
        valence = valence.view(edge_index.size(1))

        return scatter_add(
            valence, index=edge_index[0], dim=0, dim_size=num_nodes
        ).long()

    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.ligand_num_neighbors = self.count_neighbors(
            data.ligand_bond_index,
            symmetry=True,
            num_nodes=data.ligand_element.size(0),
        )
        data.ligand_atom_valence = self.count_neighbors(
            data.ligand_bond_index,
            symmetry=True,
            valence=data.ligand_bond_type,
            num_nodes=data.ligand_element.size(0),
        )
        data.ligand_atom_num_bonds = torch.stack(
            [
                self.count_neighbors(
                    data.ligand_bond_index,
                    symmetry=True,
                    valence=(data.ligand_bond_type == i).long(),
                    num_nodes=data.ligand_element.size(0),
                )
                for i in [1, 2, 3]
            ],
            dim=-1,
        )
        return data
