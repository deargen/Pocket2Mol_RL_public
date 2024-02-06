import torch
from torch import Tensor

from pocket2mol_rl.models.maskfill import MaskFillModelVN


class ToyMaskFillModelVN(MaskFillModelVN):
    def __init__(self, *args, epsilon=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def _filter_bond(self, id_element_and_bond: Tensor, index_unique: Tensor):
        """If there is no bond, remove it. Only one bond is available.

        Args:
            id_element_and_bond (Tensor):
                - id_element_and_bond[:, 0]: idx to the generated position
                - id_element_and_bond[:, 1]: element type of each atom
                - id_element_and_bond[:, 2: 2+ num_generated * n_samples]: bond type to each context atom
            index_unique (Tensor): Unique index of the generated pos, atom and bond type
        """
        # Remove the bond if there is no bond
        bond_type = id_element_and_bond[:, 2:]
        only_one_bond_index = (bond_type.sum(dim=1) == 1).nonzero(as_tuple=True)
        id_element_and_bond = id_element_and_bond[only_one_bond_index]
        index_unique = index_unique[only_one_bond_index]
        return id_element_and_bond, index_unique

    def _sample_focal_init(
        self,
        compose_feature,
        compose_pos,
        idx_protein,
        compose_knn_edge_index,
        compose_knn_edge_feature,
        frontier_threshold,  # TODO : Not used now, refactor later
    ):
        """Sample the focal atom from the frontiers.

        When sampling the focal atom from the protein frontiers,
        Sample the highest probability and apply a small epsilon.
        """
        assert self.epsilon is not None, "epsilon must be set for sampling focal atom"

        # 0: encode with fake index of ligand,
        h_compose = self._embed_compose(
            compose_feature,
            compose_pos,
            idx_ligand=torch.empty(0).to(idx_protein),
            idx_protein=idx_protein,
            compose_knn_edge_index=compose_knn_edge_index,
            compose_knn_edge_feature=compose_knn_edge_feature,
        )

        # 1: predict frontier
        y_frontier_pred = self.frontier_pred(
            h_compose,
            idx_protein,
        )[:, 0]

        # 2: get the highest probability and apply a small epsilon
        highest_prob = y_frontier_pred.max()
        ind_frontier = y_frontier_pred >= highest_prob - self.epsilon
        has_frontier = torch.any(ind_frontier)

        # 3: sample focal from frontiers
        idx_frontier = idx_protein[ind_frontier]
        p_frontier = torch.sigmoid(y_frontier_pred[ind_frontier])
        idx_focal_in_compose = torch.nonzero(ind_frontier)[:, 0]
        p_focal = p_frontier

        return (
            has_frontier,
            idx_frontier,
            p_frontier,  # frontier
            idx_focal_in_compose,
            p_focal,  # focal
            h_compose,
        )
