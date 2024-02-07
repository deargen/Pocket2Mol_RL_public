import torch
from torch import Tensor
from typing import Optional

from pocket2mol_rl.data.data import ProteinLigandData
from pocket2mol_rl.data.transform.featurize import (
    ElementFeature,
    FeaturizeLigandAtom,
    IsMolFeature,
    NumBondsFeature,
    NumNeighborsFeature,
    NumValenceFeature,
)

from pocket2mol_rl.data.action import AtomAction


def append_to_tensor(tensor: Tensor, value: Tensor, dim=0):
    """Utility function to concatenate value to a tensor along a specified dimension."""
    return torch.cat([tensor, value.to(tensor)], dim=dim)


def create_ligand_context_feature(element, feat_module: FeaturizeLigandAtom, data):
    """Generate ligand context feature tensor for the new atom."""
    feature_tensor = torch.zeros(feat_module.feature_dim).to(
        data.ligand_context_feature_full
    )

    start_idx = 0
    for feature in feat_module.feature_list:
        if isinstance(feature, ElementFeature):
            feature_tensor[start_idx : start_idx + feature.feature_dim] = (
                feature.get_one_hot(element)
            )
        elif isinstance(feature, IsMolFeature):
            feature_tensor[start_idx : start_idx + feature.feature_dim] = 1
        start_idx += feature.feature_dim

    return feature_tensor.unsqueeze(0)


def recompute_bond_related_features(data, featurize_module: FeaturizeLigandAtom):
    if data.ligand_context_size == 0:
        return data

    bond_related_features = [NumNeighborsFeature, NumValenceFeature, NumBondsFeature]

    current_idx = 0
    for feature in featurize_module.feature_list:
        if type(feature) in bond_related_features:
            if feature.feature_dim == 1:
                data.ligand_context_feature_full[:, current_idx] = feature(data)
            else:
                data.ligand_context_feature_full[
                    :, current_idx : current_idx + feature.feature_dim
                ] = feature(data)
        current_idx += feature.feature_dim

    return data


def incrementally_update_bond_related_features(
    data, bond_index, bond_type, featurize_module: FeaturizeLigandAtom
):
    """Update features related to bonds.
    Update exsisting ligand context features with the new bond.
    and add new features related to the new bond.
    """

    if len(data.ligand_context_feature_full) == 0:
        return data

    def update_num_neighbors_feature(current_idx):
        data.ligand_context_feature_full[bond_index[1, :], current_idx] += 1
        data.ligand_context_feature_full[-1, current_idx] += len(bond_index[1])

    def update_num_valence_feature(current_idx):
        data.ligand_context_feature_full[bond_index[1, :], current_idx] += bond_type
        data.ligand_context_feature_full[-1, current_idx] += torch.sum(bond_type)

    def update_num_bonds_feature(current_idx):
        data.ligand_context_feature_full[
            bond_index[1, :], current_idx + bond_type - 1
        ] += 1
        for bond in [1, 2, 3]:
            data.ligand_context_feature_full[-1, current_idx + bond - 1] += (
                bond_type == bond
            ).sum()

    feature_update_funcs = {
        NumNeighborsFeature: update_num_neighbors_feature,
        NumValenceFeature: update_num_valence_feature,
        NumBondsFeature: update_num_bonds_feature,
    }

    current_idx = 0
    for feature in featurize_module.feature_list:
        update_func = feature_update_funcs.get(type(feature))
        if update_func:
            update_func(current_idx)
        current_idx += feature.feature_dim

    return data


def get_past_data(data: ProteinLigandData, i, featurize_module: FeaturizeLigandAtom):
    """
    Args:
        data (ProteinLigandData): Data object of interest
        i (int): number of ligand atoms from the beginning to be included in the new data object.
    Returns:
        new_data (ProteinLigandData): the data object with the ligand context up to the ith atom.
    """
    n = data.ligand_context_size
    assert 0 <= i <= n

    new_data = data.clone()
    # new_data = ProteinLigandData()

    if i == n:
        return new_data

    if hasattr(data, "idx_focal"):
        new_data.idx_focal = new_data.idx_focal[:i]

    new_data.ligand_context_pos = new_data.ligand_context_pos[:i, :]
    new_data.ligand_context_element = new_data.ligand_context_element[:i]

    bond_index_mask = torch.logical_and(
        new_data.ligand_context_bond_index[0] < i,
        new_data.ligand_context_bond_index[1] < i,
    )
    new_data.ligand_context_bond_index = new_data.ligand_context_bond_index[
        :, bond_index_mask
    ]
    new_data.ligand_context_bond_type = new_data.ligand_context_bond_type[
        bond_index_mask
    ]

    new_data.ligand_context_feature_full = new_data.ligand_context_feature_full[:i, :]

    recompute_bond_related_features(new_data, featurize_module)

    return new_data


def get_past_action(
    data: ProteinLigandData, i, featurize_module: FeaturizeLigandAtom
) -> AtomAction:
    n = data.ligand_context_size
    assert 0 <= i < n

    d = {}

    if hasattr(data, "idx_focal"):
        d["idx_focal"] = data.idx_focal[i]

    d["pos"] = data.ligand_context_pos[i, :]
    d["element"] = torch.tensor(
        featurize_module.atomic_number_to_idx[data.ligand_context_element[i].item()]
    )

    bond_index_mask = torch.logical_and(
        data.ligand_context_bond_index[0] == i, data.ligand_context_bond_index[1] < i
    )
    d["bond_index"] = data.ligand_context_bond_index[:, bond_index_mask]
    d["bond_type"] = data.ligand_context_bond_type[bond_index_mask]

    return AtomAction(**d)


def get_data_with_action_applied(
    data: ProteinLigandData,
    action: AtomAction,
    featurize_module: FeaturizeLigandAtom,
):
    """Add new atom to the ligand context.
    Args:
        data (ProteinLigandData): Data object to be updated.
        action (AtomAction): Action object containing the information of the new atom.
        featurize_module (FeaturizeLigandAtom): Featurize module for ligand atoms.

    Returns:
        ProteinLigandData: Updated data object.
            - ligand_context_pos: added new position
            - ligand_context_feature_full: added new features and updated existing ligand features about bond
            - ligand_context_element: added new element
            - ligand_context_bond_index: added new bond index
            - ligand_context_bond_type: added new bond type
    """
    assert isinstance(action, AtomAction), type(action)

    return add_ligand_atom_to_data(
        data=data,
        pos=action.pos,
        element=action.element,
        bond_index=action.bond_index,
        bond_type=action.bond_type,
        featurize_module=featurize_module,
        idx_focal=action.idx_focal,
    )


def add_ligand_atom_to_data(
    data: ProteinLigandData,
    pos: Tensor,
    element: Tensor,
    bond_index: Optional[Tensor],
    bond_type: Optional[Tensor],
    featurize_module: FeaturizeLigandAtom,
    idx_focal: Optional[Tensor] = None,
):
    """Add new atom to the ligand context.
    Args:
        data (ProteinLigandData): Data object to be updated.
        pos (Tensor): Position of the new atom. Shape: (1, N) N is space dimension.
        element (Tensor): Atomic number index (as in featurize_module.atomic_numbers) of the new atom. Shape: (1,).
        bond_index (Tensor): Index of the new bond. Shape: (2, M) M is number of bonds.
        bond_type (Tensor): Type of the new bond. Shape: (M,).
        featurize_module (FeaturizeLigandAtom): Featurize module for ligand atoms.

    Returns:
        ProteinLigandData: Updated data object.
            - ligand_context_pos: added new position
            - ligand_context_feature_full: added new features and updated existing ligand features about bond
            - ligand_context_element: added new element
            - ligand_context_bond_index: added new bond index
            - ligand_context_bond_type: added new bond type
    """

    data = data.clone()

    if idx_focal is not None:
        if not hasattr(data, "idx_focal"):
            data.idx_focal = torch.empty((0,))
        data.idx_focal = append_to_tensor(data.idx_focal, idx_focal.view(1))

    if hasattr(data, "ligand_focalizable_mask"):
        data.ligand_focalizable_mask = append_to_tensor(
            data.ligand_focalizable_mask, torch.tensor([True]).view(1)
        )
    if hasattr(data, "ligand_branch_point_mask"):
        data.ligand_branch_point_mask = append_to_tensor(
            data.ligand_branch_point_mask, torch.tensor([False]).view(1)
        )

    # Update ligand context position
    data.ligand_context_pos = append_to_tensor(data.ligand_context_pos, pos.view(1, -1))

    # Update ligand context features
    ligand_context_feature = create_ligand_context_feature(
        element, featurize_module, data
    )
    data.ligand_context_feature_full = append_to_tensor(
        data.ligand_context_feature_full, ligand_context_feature
    )

    # Update ligand context element
    element_mapped = torch.LongTensor([featurize_module.atomic_numbers[element.item()]])
    data.ligand_context_element = append_to_tensor(
        data.ligand_context_element, element_mapped.view(1)
    )

    if bond_type is not None and len(bond_type) != 0:
        # If there is a bond, update bond indexes and types
        bond_index[0, :] = (
            len(data.ligand_context_pos) - 1
        )  # Before this change, elements of bond_index[0] are query*sample indices

        # Symmetrizing bond indexes and types
        bond_index_all = torch.cat([bond_index, bond_index.flip([0])], dim=1)
        bond_type_all = torch.cat([bond_type, bond_type], dim=0)

        # Updating bond indexes and types
        data.ligand_context_bond_index = append_to_tensor(
            data.ligand_context_bond_index, bond_index_all, dim=1
        )
        data.ligand_context_bond_type = append_to_tensor(
            data.ligand_context_bond_type, bond_type_all
        )

        # Update ligand context features related to bonds
        incrementally_update_bond_related_features(
            data, bond_index, bond_type, featurize_module
        )
    return data
