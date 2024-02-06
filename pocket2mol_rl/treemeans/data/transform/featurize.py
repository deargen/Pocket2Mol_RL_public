from typing import List

import torch

from pocket2mol_rl.data.data import ProteinLigandData
from pocket2mol_rl.data.transform import (
    ElementFeature,
    FeaturizeLigandAtom,
    FeaturizeProteinAtom,
    LigandFeature,
    NumNeighborsFeature,
)


class ToyFeaturizeProteinAtom(FeaturizeProteinAtom):
    def __init__(self):
        """Add protein atom feature to ProteinLigandData
        In this toy task, we only have one type of atom in the protein

        Attributes:
            atomic_numbers (torch.LongTensor): The atomic number of the atoms in the protein
        """
        super().__init__()
        self.atomic_numbers = torch.LongTensor([0])

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0)

    def __call__(self, data: ProteinLigandData):
        element = data.protein_element.view(-1, 1) == self.atomic_numbers.view(
            1, -1
        )  # (N_atoms, N_elements)
        data.protein_atom_feature = element.long()
        return data


class ToyElementFeature(ElementFeature):
    def __init__(self):
        self.atomic_numbers = torch.LongTensor([0])


class ToyFeaturizeLigandAtom(FeaturizeLigandAtom):
    def __init__(self):
        """Add ligand atom feature to ProteinLigandData
        In this toy task, we only have one type of atom in the ligand
        Additionally, we add the number of neighbors only. Because we only have one type of bond.

        Attributes:
            atomic_numbers (torch.LongTensor): The atomic number of the atoms in the ligand
        """
        super().__init__()
        self.feature_list: List[LigandFeature] = [
            ToyElementFeature(),
            NumNeighborsFeature(),
        ]
        self.atomic_numbers: torch.Tensor = ToyElementFeature().atomic_numbers

    def __call__(self, data: ProteinLigandData):
        element = data.ligand_element.view(-1, 1) == self.atomic_numbers.view(
            1, -1
        )  # (N_atoms, N_elements)
        n_neigh = data.ligand_num_neighbors.view(-1, 1)
        x = torch.cat([element, n_neigh], dim=-1)
        data.ligand_atom_feature_full = x
        return data
