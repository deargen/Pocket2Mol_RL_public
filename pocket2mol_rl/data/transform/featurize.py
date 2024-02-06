from typing import List, Dict, Optional

import torch
import torch.nn.functional as F

from pocket2mol_rl.data.data import ProteinLigandData
from .count_neighbors import LigandCountNeighbors


class FeaturizeProteinAtom(object):
    def __init__(self):
        super().__init__()
        self.atomic_numbers = torch.LongTensor([6, 7, 8, 16, 34])  # C, N, O, S, Se
        self.max_num_aa = 20

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1 + 1

    def __call__(self, data: ProteinLigandData):
        element = data.protein_element.view(-1, 1) == self.atomic_numbers.view(
            1, -1
        )  # (N_atoms, N_elements)
        amino_acid = F.one_hot(
            data.protein_atom_to_aa_type, num_classes=self.max_num_aa
        )
        is_backbone = data.protein_is_backbone.view(-1, 1).long()
        is_mol_atom = torch.zeros_like(is_backbone, dtype=torch.long)
        # x = torch.cat([element, amino_acid, is_backbone], dim=-1)
        x = torch.cat([element, amino_acid, is_backbone, is_mol_atom], dim=-1)
        data.protein_atom_feature = x
        # data.compose_index = torch.arange(len(element), dtype=torch.long)
        return data


class LigandFeature(object):
    @property
    def feature_dim(self) -> int:
        raise NotImplementedError

    @property
    def to_change_after_masking(self) -> bool:
        """Whether the feature needs to be changed after masking transformation."""
        raise NotImplementedError

    def attr_after_change(self) -> str:
        """Add the changed feature to the data.
        If to_change_after_masking is False, it is not used.
        """
        raise NotImplementedError("if to_change_after_masking is True, implement it")

    def __call__(self, data: ProteinLigandData) -> torch.Tensor:
        """Calculate the changed feature.
        If to_change_after_masking is False, it is not used."""
        raise NotImplementedError("if to_change_after_masking is True, implement it")


class ElementFeature(LigandFeature):
    def __init__(self):
        # C N O F P S Cl
        self.atomic_numbers = torch.LongTensor([6, 7, 8, 9, 15, 16, 17])

        # typical in organic molecules
        self.max_possible_valences = {
            6: 4,  # C
            7: 3,  # N
            8: 2,  # O
            9: 1,  # F
            15: 5,  # P
            16: 6,  # S
            17: 1,  # Cl
        }

    @property
    def feature_dim(self) -> int:
        return self.atomic_numbers.size(0)

    @property
    def to_change_after_masking(self) -> bool:
        return False

    def get_one_hot(self, element: torch.Tensor) -> torch.Tensor:
        """Get one-hot encoding of the element. Element is a tensor of shape (1).
        And index of the element is the same as the index of the one-hot encoding.
        (e.g. C is 0, N is 1, O is 2, ...)
        """
        return F.one_hot(element.view(-1), num_classes=self.atomic_numbers.size(0))


class IsMolFeature(LigandFeature):
    @property
    def feature_dim(self) -> int:
        return 1

    @property
    def to_change_after_masking(self) -> bool:
        return False


class NumNeighborsFeature(LigandFeature):
    @property
    def feature_dim(self) -> int:
        return 1

    @property
    def to_change_after_masking(self) -> bool:
        return True

    def attr_after_change(self) -> str:
        return "ligand_context_num_neighbors"

    def __call__(self, data: ProteinLigandData) -> torch.Tensor:
        return LigandCountNeighbors.count_neighbors(
            data.ligand_context_bond_index,
            symmetry=True,
            num_nodes=data.ligand_context_size,
        )


class NumValenceFeature(LigandFeature):
    @property
    def feature_dim(self) -> int:
        return 1

    @property
    def to_change_after_masking(self) -> bool:
        return True

    def attr_after_change(self) -> str:
        return "ligand_context_valence"

    def __call__(self, data: ProteinLigandData) -> torch.Tensor:
        return LigandCountNeighbors.count_neighbors(
            data.ligand_context_bond_index,
            symmetry=True,
            valence=data.ligand_context_bond_type,
            num_nodes=data.ligand_context_size,
        )


class NumBondsFeature(LigandFeature):
    @property
    def feature_dim(self) -> int:
        return 3

    @property
    def to_change_after_masking(self) -> bool:
        return True

    def attr_after_change(self) -> str:
        return "ligand_context_num_bonds"

    def __call__(self, data: ProteinLigandData) -> torch.Tensor:
        return torch.stack(
            [
                LigandCountNeighbors.count_neighbors(
                    data.ligand_context_bond_index,
                    symmetry=True,
                    valence=(data.ligand_context_bond_type == i).long(),
                    num_nodes=data.ligand_context_size,
                )
                for i in [1, 2, 3]
            ],
            dim=-1,
        )


class FeaturizeLigandAtom(object):
    def __init__(self):
        """Featurize ligand atom.

        Attributes:
            feature_list (List[LigandFeature]): list of ligand features,
                feature list is used in change_features_of_neigh which is used in masking transformation,
                First feature in the list should be ElementFeature, which is to use get_one_hot on sampling.
            atomic_numbers (torch.Tensor): atomic numbers of ligand atoms, which is defined in ElementFeature
        """
        super().__init__()
        self.feature_list: List[LigandFeature] = [
            ElementFeature(),
            IsMolFeature(),
            NumNeighborsFeature(),
            NumValenceFeature(),
            NumBondsFeature(),
        ]
        self.atomic_numbers: torch.Tensor = ElementFeature().atomic_numbers
        self.max_possible_valences: Dict[
            int, int
        ] = ElementFeature().max_possible_valences
        self.atomic_number_to_idx = {
            self.atomic_numbers[i].item(): i for i in range(len(self.atomic_numbers))
        }

    @property
    def feature_dim(self) -> int:
        feature_dim = 0
        for feature in self.feature_list:
            feature_dim += feature.feature_dim
        return feature_dim

    def __call__(self, data: ProteinLigandData):
        element = data.ligand_element.view(-1, 1) == self.atomic_numbers.view(
            1, -1
        )  # (N_atoms, N_elements)
        is_mol_atom = torch.ones([len(element), 1], dtype=torch.long)
        n_neigh = data.ligand_num_neighbors.view(-1, 1)
        n_valence = data.ligand_atom_valence.view(-1, 1)
        ligand_atom_num_bonds = data.ligand_atom_num_bonds
        x = torch.cat(
            [element, is_mol_atom, n_neigh, n_valence, ligand_atom_num_bonds], dim=-1
        )
        data.ligand_atom_feature_full = x
        return data

    def change_features_of_neigh(self, data: ProteinLigandData) -> ProteinLigandData:
        """
        Used in masking transformation. Change the features of the context atom.

        Args:
            data (ProteinLigandData): data containing ligand_context_feature_full and related features

        Returns:
            ProteinLigandData: data with changed ligand_feature_full
        """
        current_idx = 0
        for feature in self.feature_list:
            idx = current_idx + feature.feature_dim
            if feature.to_change_after_masking:
                changed_feature = feature(data).long()

                # Add changed feature to data
                setattr(data, feature.attr_after_change(), changed_feature)

                # Change ligand_context_feature_full of the context atom
                if changed_feature.dim() == 1:
                    changed_feature = changed_feature.view(-1, 1)
                data.ligand_context_feature_full[:, current_idx:idx] = changed_feature
            current_idx = idx
        return data
