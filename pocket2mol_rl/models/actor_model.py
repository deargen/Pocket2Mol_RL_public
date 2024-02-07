from typing import Any, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import knn

from pocket2mol_rl.data.action import (
    Action,
    ActionConditionalDistribution,
    AtomAction,
    BondDistribution,
    ElementDistribution,
    FocalStopAction,
    IdxFocalDistribution,
    PosDistribution,
    StopAction,
)
from pocket2mol_rl.data.data import ProteinLigandData
from pocket2mol_rl.models.pretrained_maskfill import PretrainedMaskFillModel
from pocket2mol_rl.utils.tensor import (
    added_concat,
    decompose_tensors,
    decompose_tensors_and_getindex,
)


class ProbabilityMixIn(PretrainedMaskFillModel):
    """ProbabilityMixIn provides methods for computing the log probabilities
    and conditional distributions of the actions.
    To be used as a mixin in [`MaskFillModelVN`].
    """

    def set_focal_action_stop_first(self, focal_action_stop_first):
        assert isinstance(focal_action_stop_first, bool)
        self.focal_action_stop_first = focal_action_stop_first

    @staticmethod
    def _get_batch_attrs(data_list, device):
        """
        Extracts relevant attributes from a list of data and returns them as a tuple.

        Args:
            data_list (list): A list of data.
            device (torch.device): The device to move the extracted attributes to.

        Returns:
            tuple: A tuple containing the following attributes:
                compose_feature (torch.Tensor): A tensor of shape (batch_size, num_features) containing the composition features.
                compose_pos (torch.Tensor): A tensor of shape (num_nodes, num_dimensions) containing the composition positions.
                idx_ligand (torch.Tensor): A tensor of shape (num_ligands,) containing the indices of the ligands.
                idx_protein (torch.Tensor): A tensor of shape (num_proteins,) containing the indices of the proteins.
                compose_knn_edge_index (torch.Tensor): A tensor of shape (2, num_edges) containing the indices of the composition edges.
                compose_knn_edge_feature (torch.Tensor): A tensor of shape (num_edges, num_features) containing the features of the composition edges.
                ligand_context_bond_index (torch.Tensor): A tensor of shape (num_ligand_context_bonds, 2) containing the indices of the ligand context bonds.
                ligand_context_bond_type (torch.Tensor): A tensor of shape (num_ligand_context_bonds,) containing the types of the ligand context bonds.
        """

        batch = Batch.from_data_list(data_list, exclude_keys=["ligand_nbh_list"]).to(
            device
        )
        compose_feature = batch.compose_feature.float()
        compose_pos = batch.compose_pos
        idx_ligand = batch.idx_ligand_ctx_in_compose
        idx_protein = batch.idx_protein_in_compose
        compose_knn_edge_index = batch.compose_knn_edge_index
        compose_knn_edge_feature = batch.compose_knn_edge_feature
        ligand_context_bond_index = batch.ligand_context_bond_index
        ligand_context_bond_type = batch.ligand_context_bond_type
        return (
            compose_feature,
            compose_pos,
            idx_ligand,
            idx_protein,
            compose_knn_edge_index,
            compose_knn_edge_feature,
            ligand_context_bond_index,
            ligand_context_bond_type,
        )

    def compute_probs_off_policy_in_batch(
        self, data_list: List[Any], actions: List[Action]
    ):
        # Ensure all actions and data are moved to the same device as model's parameters.
        data_device = data_list[0].compose_pos.device
        device = next(self.parameters()).device
        action_device = actions[0].device
        actions = [action.to(device) for action in actions]

        # Prepare the batched attributes from data.
        (
            compose_feature,
            compose_pos,
            idx_ligand,
            idx_protein,
            compose_knn_edge_index,
            compose_knn_edge_feature,
            ligand_context_bond_index,
            ligand_context_bond_type,
        ) = self._get_batch_attrs(data_list, device)

        # Identify if the context size is initial for all data points.
        compose_sizes = [len(data.compose_pos) for data in data_list]
        ligand_context_sizes = [
            len(data.idx_ligand_ctx_in_compose) for data in data_list
        ]
        is_init = [size == 0 for size in ligand_context_sizes]

        # This is sufficient for now. This assumption simplifies the element and bond prediction.
        assert all(is_init) or not any(is_init)
        if all(is_init):
            idx_ligand = torch.empty(0).to(idx_protein)

        focal_candidate_idxs_list = [
            (
                data.idx_protein_in_compose.to(device)
                if is_init[i]
                else data.idx_ligand_ctx_in_compose.to(device)
            )
            for i, data in enumerate(data_list)
        ]
        focal_candidate_idxs_in_compose = added_concat(
            focal_candidate_idxs_list, compose_sizes, dim=0
        )
        focal_candidate_sizes = [len(idxs) for idxs in focal_candidate_idxs_list]

        logps = [None for _ in range(len(data_list))]
        pis = [None for _ in range(len(data_list))]
        pi_idx_focal_list = [None for _ in range(len(data_list))]
        pi_pos_list = [None for _ in range(len(data_list))]
        pi_element_list = [None for _ in range(len(data_list))]
        pi_bond_list = [None for _ in range(len(data_list))]

        h_compose = self._embed_compose(
            compose_feature,
            compose_pos,
            idx_ligand,
            idx_protein,
            compose_knn_edge_index,
            compose_knn_edge_feature,
        )

        y_frontier_pred = self.frontier_pred(
            h_compose, focal_candidate_idxs_in_compose
        )[:, 0]
        y_frontier_pred_list = decompose_tensors(
            y_frontier_pred, focal_candidate_sizes, dim=0
        )

        focalizable_mask_list = [None for _ in range(len(data_list))]
        for i, data in enumerate(data_list):
            if hasattr(data, "ligand_focalizable_mask"):
                assert not is_init[i]
                focalizable_mask_list[i] = data.ligand_focalizable_mask.to(device)

        for i, y in enumerate(y_frontier_pred_list):
            pi_idx_focal_list[i] = IdxFocalDistribution(
                y,
                allow_stop=not is_init[i],
                stop_first=self.focal_action_stop_first,
                focalizable_mask=focalizable_mask_list[i],
            )

        for i, action in enumerate(actions):
            if isinstance(action, StopAction):
                assert isinstance(action, FocalStopAction)
                pi = ActionConditionalDistribution(
                    condition=action,
                    pi_idx_focal=pi_idx_focal_list[i],
                    pi_pos=None,
                    pi_element=None,
                    pi_bond=None,
                )
                pis[i] = pi

                logp = pi.get_logp(action)
                logps[i] = logp

        remaining_idxs = [
            i for i, action in enumerate(actions) if not isinstance(action, StopAction)
        ]
        if len(remaining_idxs) == 0:
            assert all(logp is not None for logp in logps)
            assert all(pi is not None for pi in pis)
            return logps, pis

        # Retain only the remaining data
        assert type(h_compose) == list and len(h_compose) == 2
        h_composes = [
            decompose_tensors(h_compose[0], compose_sizes, dim=0),
            decompose_tensors(h_compose[1], compose_sizes, dim=0),
        ]
        h_compose = [
            torch.cat([h_composes[0][idx] for idx in remaining_idxs], dim=0),
            torch.cat([h_composes[1][idx] for idx in remaining_idxs], dim=0),
        ]
        remaining_data_list = [data_list[idx] for idx in remaining_idxs]

        (
            compose_feature,
            compose_pos,
            idx_ligand,
            idx_protein,
            compose_knn_edge_index,
            compose_knn_edge_feature,
            ligand_context_bond_index,
            ligand_context_bond_type,
        ) = self._get_batch_attrs(remaining_data_list, device)

        remaining_compose_sizes = [compose_sizes[idx] for idx in remaining_idxs]
        remaining_ligand_context_sizes = [
            ligand_context_sizes[idx] for idx in remaining_idxs
        ]
        remaining_is_init = [is_init[idx] for idx in remaining_idxs]

        # Compute pi_pos
        idx_focal_in_compose = added_concat(
            [actions[idx].idx_focal.view(1).to(device) for idx in remaining_idxs],
            remaining_compose_sizes,
            dim=0,
        )
        assert len(idx_focal_in_compose.shape) == 1
        relative_pos_mu, abs_pos_mus, pos_sigmas, pos_pis = self.pos_predictor(
            h_compose, idx_focal_in_compose, compose_pos
        )
        assert (
            len(abs_pos_mus) == len(pos_sigmas) == len(pos_pis) == len(remaining_idxs)
        )
        for i, idx in enumerate(remaining_idxs):
            abs_pos_mu = abs_pos_mus[i]
            pos_sigma = pos_sigmas[i]
            pos_pi = pos_pis[i]
            pi_pos = PosDistribution(abs_pos_mu, pos_sigma, pos_pi)
            pi_pos_list[idx] = pi_pos

        # Compute pi_element and pi_bond
        def _get_query_compose_knn_edge_index():
            query_compose_knn_edge_index_i_list = []
            query_compose_knn_edge_index_j_list = []
            compose_poses = decompose_tensors(
                compose_pos, remaining_compose_sizes, dim=0
            )  # to knn only within the same data
            for i, idx in enumerate(remaining_idxs):
                query_compose_knn_edge_index_j = knn(
                    x=compose_poses[i],
                    y=pos_query[i].unsqueeze(0),
                    k=self.config.field.knn,
                    num_workers=16,
                )
                assert torch.all(query_compose_knn_edge_index_j[0] == 0)
                query_compose_knn_edge_index_j = query_compose_knn_edge_index_j[1]
                query_compose_knn_edge_index_i_list.append(
                    i * torch.ones_like(query_compose_knn_edge_index_j)
                )
                query_compose_knn_edge_index_j_list.append(
                    query_compose_knn_edge_index_j
                )
            query_compose_knn_edge_index_i = torch.cat(
                query_compose_knn_edge_index_i_list, dim=0
            )
            """
            query_compose_knn_edge_index_j = torch.cat(
                query_compose_knn_edge_index_j_list, dim=0
            )
            """
            query_compose_knn_edge_index_j = added_concat(
                query_compose_knn_edge_index_j_list, remaining_compose_sizes, dim=0
            )
            return torch.stack(
                [query_compose_knn_edge_index_i, query_compose_knn_edge_index_j], dim=0
            )

        assert all(isinstance(actions[idx], AtomAction) for idx in remaining_idxs)
        pos_query = torch.stack([actions[idx].pos for idx in remaining_idxs], dim=0)
        query_compose_knn_edge_index = _get_query_compose_knn_edge_index()

        if all(remaining_is_init):
            y_query_preds, _ = self.field(
                pos_query=pos_query,
                edge_index_query=[],
                pos_compose=compose_pos,
                node_attr_compose=h_compose,
                edge_index_q_cps_knn=query_compose_knn_edge_index,
            )
        else:
            assert not any(
                remaining_is_init
            )  # Due to the assumption `all(is_init) or not any(is_init)`
            edge_index_query_i = []
            for i, size in enumerate(remaining_ligand_context_sizes):
                edge_index_query_i.extend([i] * size)
            edge_index_query_i = torch.tensor(edge_index_query_i, device=device)
            assert len(edge_index_query_i) == len(idx_ligand)
            edge_index_query_j = added_concat(
                [
                    torch.arange(size, device=device)
                    for size in remaining_ligand_context_sizes
                ],
                remaining_compose_sizes,
                dim=0,
            )
            edge_index_query = torch.stack(
                [edge_index_query_i, edge_index_query_j], dim=0
            )

            (
                index_real_cps_edge_for_atten,
                tri_edge_index,
                tri_edge_feat,
            ) = self._get_tri_edges(
                edge_index_query=edge_index_query,
                pos_query=pos_query,
                idx_ligand=idx_ligand,
                ligand_bond_index=ligand_context_bond_index,
                ligand_bond_type=ligand_context_bond_type,
                modified=True,
            )
            y_query_preds, edge_preds = self.field(
                pos_query=pos_query,
                edge_index_query=edge_index_query,
                pos_compose=compose_pos,
                node_attr_compose=h_compose,
                edge_index_q_cps_knn=query_compose_knn_edge_index,
                index_real_cps_edge_for_atten=index_real_cps_edge_for_atten,
                tri_edge_index=tri_edge_index,
                tri_edge_feat=tri_edge_feat,
            )

        assert len(y_query_preds.shape) == 2 and y_query_preds.shape[0] == len(
            remaining_idxs
        )
        if not all(remaining_is_init):
            assert edge_preds.shape == (len(idx_ligand), self.num_edge_types), (
                edge_preds.shape,
                (len(idx_ligand), self.num_edge_types),
            )
        for i, idx in enumerate(remaining_idxs):
            y_query_pred = y_query_preds[i]
            pi_element = ElementDistribution(y_query_pred)
            pi_element_list[idx] = pi_element

            if not all(remaining_is_init):
                edge_pred = decompose_tensors_and_getindex(
                    edge_preds, remaining_ligand_context_sizes, i, dim=0
                )
                pi_bond = BondDistribution(edge_pred)
                pi_bond_list[idx] = pi_bond

        # Wrap up the results
        for i, idx in enumerate(remaining_idxs):
            assert pi_idx_focal_list[idx] is not None
            assert pi_pos_list[idx] is not None
            assert pi_element_list[idx] is not None
            if remaining_is_init[i]:
                assert pi_bond_list[idx] is None
            else:
                assert pi_bond_list[idx] is not None

            pi = ActionConditionalDistribution(
                condition=actions[idx],
                pi_idx_focal=pi_idx_focal_list[idx],
                pi_pos=pi_pos_list[idx],
                pi_element=pi_element_list[idx],
                pi_bond=pi_bond_list[idx],
            )
            pis[idx] = pi

            logp = pi.get_logp(actions[idx])
            logps[idx] = logp

        assert all(logp is not None for logp in logps)
        assert all(pi is not None for pi in pis)
        return logps, pis

    def compute_probs_off_policy(
        self,
        data: ProteinLigandData,
        action: Action,
    ) -> Tuple[Tensor, ActionConditionalDistribution]:
        """Get the log probabilites and conditional probabilites for an off-policy (sampled in a past episode) data-action pair, with respect to the current model."""
        assert action is not None

        data_device = data.compose_pos.device
        device = next(self.parameters()).device
        batch = Batch.from_data_list([data]).to(device)
        compose_feature = batch.compose_feature.float()
        compose_pos = batch.compose_pos
        idx_ligand = batch.idx_ligand_ctx_in_compose
        idx_protein = batch.idx_protein_in_compose
        compose_knn_edge_index = batch.compose_knn_edge_index
        compose_knn_edge_feature = batch.compose_knn_edge_feature
        ligand_context_bond_index = batch.ligand_context_bond_index
        ligand_context_bond_type = batch.ligand_context_bond_type

        action_device = action.device
        action = action.to(device)

        is_init = len(idx_ligand) == 0

        if is_init:
            pi_idx_focal, h_compose = self._compute_init_pi_idx_focal(
                compose_feature,
                compose_pos,
                idx_protein,
                compose_knn_edge_index,
                compose_knn_edge_feature,
            )
        else:
            if hasattr(data, "ligand_focalizable_mask"):
                focalizable_mask = data.ligand_focalizable_mask.to(device)
            else:
                focalizable_mask = None
            pi_idx_focal, h_compose = self._compute_pi_idx_focal(
                compose_feature,
                compose_pos,
                idx_ligand,
                idx_protein,
                compose_knn_edge_index,
                compose_knn_edge_feature,
                focalizable_mask=focalizable_mask,
            )

        if isinstance(action, StopAction):
            assert isinstance(action, FocalStopAction)
            pi = ActionConditionalDistribution(
                condition=action,
                pi_idx_focal=pi_idx_focal,
                pi_pos=None,
                pi_element=None,
                pi_bond=None,
            )
            logp = pi.get_logp(action)

            return (logp, pi)

        pi_pos = self._compute_pi_pos(h_compose, compose_pos, action.idx_focal)

        if is_init:
            pi_element = self._compute_init_pi_element(
                action.pos,
                h_compose,
                compose_pos,
            )
            pi_bond = None
        else:
            pi_element, pi_bond = self._compute_pi_element_bond(
                action.pos,
                h_compose,
                compose_pos,
                idx_ligand,
                ligand_context_bond_index,
                ligand_context_bond_type,
            )

        pi = ActionConditionalDistribution(
            condition=action,
            pi_idx_focal=pi_idx_focal,
            pi_pos=pi_pos,
            pi_element=pi_element,
            pi_bond=pi_bond,
        )
        logp = pi.get_logp(action)

        action = action.to(action_device)

        return (logp, pi)

    def sample_one_in_batch(
        self,
        data_list: List[Any],
        pos_only_mean=False,
        hard_stop=None,
        seed=None,
        generator=None,
        return_activation_stat=False,
    ):
        if return_activation_stat:
            act_stats = [{} for _ in range(len(data_list))]

        def _get_batch_attrs(data_list, device):
            batch = Batch.from_data_list(
                data_list, exclude_keys=["ligand_nbh_list"]
            ).to(device)
            compose_feature = batch.compose_feature.float()
            compose_pos = batch.compose_pos
            idx_ligand = batch.idx_ligand_ctx_in_compose
            idx_protein = batch.idx_protein_in_compose
            compose_knn_edge_index = batch.compose_knn_edge_index
            compose_knn_edge_feature = batch.compose_knn_edge_feature
            ligand_context_bond_index = batch.ligand_context_bond_index
            ligand_context_bond_type = batch.ligand_context_bond_type
            return (
                compose_feature,
                compose_pos,
                idx_ligand,
                idx_protein,
                compose_knn_edge_index,
                compose_knn_edge_feature,
                ligand_context_bond_index,
                ligand_context_bond_type,
            )

        data_device = data_list[0].compose_pos.device
        device = next(self.parameters()).device
        (
            compose_feature,
            compose_pos,
            idx_ligand,
            idx_protein,
            compose_knn_edge_index,
            compose_knn_edge_feature,
            ligand_context_bond_index,
            ligand_context_bond_type,
        ) = _get_batch_attrs(data_list, device)

        compose_sizes = [len(data.compose_pos) for data in data_list]
        ligand_context_sizes = [
            len(data.idx_ligand_ctx_in_compose) for data in data_list
        ]
        is_init = [size == 0 for size in ligand_context_sizes]
        assert all(is_init) or not any(
            is_init
        )  # This is sufficient for now. This assumption simplifies the element and bond prediction.
        if all(is_init):
            idx_ligand = torch.empty(0).to(idx_protein)
        focal_candidate_idxs_list = [
            (
                data.idx_protein_in_compose.to(device)
                if is_init[i]
                else data.idx_ligand_ctx_in_compose.to(device)
            )
            for i, data in enumerate(data_list)
        ]
        focal_candidate_idxs_in_compose = added_concat(
            focal_candidate_idxs_list, compose_sizes, dim=0
        )
        focal_candidate_sizes = [len(idxs) for idxs in focal_candidate_idxs_list]

        logps = [None for _ in range(len(data_list))]
        pis = [None for _ in range(len(data_list))]
        pi_idx_focal_list = [None for _ in range(len(data_list))]
        pi_pos_list = [None for _ in range(len(data_list))]
        pi_element_list = [None for _ in range(len(data_list))]
        pi_bond_list = [None for _ in range(len(data_list))]

        actions = [None for _ in range(len(data_list))]
        idx_focal_list = [None for _ in range(len(data_list))]
        pos_list = [None for _ in range(len(data_list))]
        element_list = [None for _ in range(len(data_list))]
        bond_list = [None for _ in range(len(data_list))]

        h_compose = self._embed_compose(
            compose_feature,
            compose_pos,
            idx_ligand,
            idx_protein,
            compose_knn_edge_index,
            compose_knn_edge_feature,
        )

        y_frontier_pred = self.frontier_pred(
            h_compose, focal_candidate_idxs_in_compose
        )[:, 0]
        y_frontier_pred_list = decompose_tensors(
            y_frontier_pred, focal_candidate_sizes, dim=0
        )

        focalizable_mask_list = [None for _ in range(len(data_list))]
        for i, data in enumerate(data_list):
            if hasattr(data, "ligand_focalizable_mask"):
                assert not is_init[i]
                focalizable_mask_list[i] = data.ligand_focalizable_mask.to(device)

        for i, y in enumerate(y_frontier_pred_list):
            pi_idx_focal = IdxFocalDistribution(
                y,
                allow_stop=not is_init[i],
                stop_first=self.focal_action_stop_first,
                focalizable_mask=focalizable_mask_list[i],
            )
            pi_idx_focal_list[i] = pi_idx_focal

            idx_focal = pi_idx_focal.sample(
                hard_stop=hard_stop, seed=seed, generator=generator
            )
            idx_focal_list[i] = idx_focal

            if idx_focal is None:
                action = FocalStopAction()
                actions[i] = action

                pi = ActionConditionalDistribution(
                    condition=action,
                    pi_idx_focal=pi_idx_focal,
                    pi_pos=None,
                    pi_element=None,
                    pi_bond=None,
                )
                pis[i] = pi

                logp = pi.get_logp(action)
                logps[i] = logp

        if return_activation_stat:
            for i, y in enumerate(y_frontier_pred_list):
                act_stats[i]["focal_stop"] = isinstance(actions[i], FocalStopAction)
                act_stats[i]["focal_logit_n"] = len(y)
                act_stats[i]["focal_logit_sum"] = y.sum().item()
                act_stats[i]["focal_logit_sq_sum"] = (y**2).sum().item()

        remaining_idxs = [
            i for i, action in enumerate(actions) if not isinstance(action, StopAction)
        ]
        if len(remaining_idxs) == 0:
            assert all(action is not None for action in actions)
            assert all(logp is not None for logp in logps)
            assert all(pi is not None for pi in pis)

            for i, action in enumerate(actions):
                actions[i] = action.to(data_device)

            if return_activation_stat:
                return actions, (logps, pis), act_stats
            else:
                return actions, (logps, pis)

        # Retain only the remaining data
        assert type(h_compose) == list and len(h_compose) == 2
        h_composes = [
            decompose_tensors(h_compose[0], compose_sizes, dim=0),
            decompose_tensors(h_compose[1], compose_sizes, dim=0),
        ]
        h_compose = [
            torch.cat([h_composes[0][idx] for idx in remaining_idxs], dim=0),
            torch.cat([h_composes[1][idx] for idx in remaining_idxs], dim=0),
        ]
        remaining_data_list = [data_list[idx] for idx in remaining_idxs]
        (
            compose_feature,
            compose_pos,
            idx_ligand,
            idx_protein,
            compose_knn_edge_index,
            compose_knn_edge_feature,
            ligand_context_bond_index,
            ligand_context_bond_type,
        ) = _get_batch_attrs(remaining_data_list, device)
        remaining_compose_sizes = [compose_sizes[idx] for idx in remaining_idxs]
        remaining_ligand_context_sizes = [
            ligand_context_sizes[idx] for idx in remaining_idxs
        ]
        remaining_is_init = [is_init[idx] for idx in remaining_idxs]

        # Compute pi_pos
        assert all(idx_focal_list[idx] is not None for idx in remaining_idxs)
        idx_focal_in_compose = added_concat(
            [idx_focal_list[idx].view(1).to(device) for idx in remaining_idxs],
            remaining_compose_sizes,
            dim=0,
        )
        assert len(idx_focal_in_compose.shape) == 1
        relative_pos_mu, abs_pos_mus, pos_sigmas, pos_pis = self.pos_predictor(
            h_compose, idx_focal_in_compose, compose_pos
        )
        assert (
            len(abs_pos_mus) == len(pos_sigmas) == len(pos_pis) == len(remaining_idxs)
        )
        for i, idx in enumerate(remaining_idxs):
            abs_pos_mu = abs_pos_mus[i]
            pos_sigma = pos_sigmas[i]
            pos_pi = pos_pis[i]
            pi_pos = PosDistribution(abs_pos_mu, pos_sigma, pos_pi)
            pi_pos_list[idx] = pi_pos

            pos = pi_pos.sample(only_mean=pos_only_mean, seed=seed, generator=generator)
            pos_list[idx] = pos

            if return_activation_stat:
                pos_to_generate = self.pos_predictor.get_maximum(
                    abs_pos_mu, pos_sigma, pos_pi
                )

                """ 
                pdf_pos = self.pos_predictor.get_mdn_probability(
                    mu=abs_pos_mu,
                    sigma=pos_sigma,
                    pi=pos_pi,
                    pos_target=pos_to_generate,
                )
                """
                pdf_pos = pi_pos.get_pdf(pos_to_generate)

                # top-5 indices in pos_pi
                num = min(5, len(pos_pi))
                assert num > 0
                top_5_idxs = torch.argsort(pos_pi, descending=True)[:num]

                act_stats[idx]["top-5_mu_pdf_mean"] = pdf_pos[top_5_idxs].mean().item()
                act_stats[idx]["mu_pdf_mean"] = pdf_pos.mean().item()

        # Compute pi_element and pi_bond
        def _get_query_compose_knn_edge_index():
            query_compose_knn_edge_index_i_list = []
            query_compose_knn_edge_index_j_list = []
            compose_poses = decompose_tensors(
                compose_pos, remaining_compose_sizes, dim=0
            )  # to knn only within the same data
            for i, idx in enumerate(remaining_idxs):
                query_compose_knn_edge_index_j = knn(
                    x=compose_poses[i],
                    y=pos_query[i].unsqueeze(0),
                    k=self.config.field.knn,
                    num_workers=16,
                )
                assert torch.all(query_compose_knn_edge_index_j[0] == 0)
                query_compose_knn_edge_index_j = query_compose_knn_edge_index_j[1]
                query_compose_knn_edge_index_i_list.append(
                    i * torch.ones_like(query_compose_knn_edge_index_j)
                )
                query_compose_knn_edge_index_j_list.append(
                    query_compose_knn_edge_index_j
                )
            query_compose_knn_edge_index_i = torch.cat(
                query_compose_knn_edge_index_i_list, dim=0
            )
            """
            query_compose_knn_edge_index_j = torch.cat(
                query_compose_knn_edge_index_j_list, dim=0
            )
            """
            query_compose_knn_edge_index_j = added_concat(
                query_compose_knn_edge_index_j_list, remaining_compose_sizes, dim=0
            )
            return torch.stack(
                [query_compose_knn_edge_index_i, query_compose_knn_edge_index_j], dim=0
            )

        assert all(pos_list[idx] is not None for idx in remaining_idxs)
        pos_query = torch.stack([pos_list[idx] for idx in remaining_idxs], dim=0)
        query_compose_knn_edge_index = _get_query_compose_knn_edge_index()

        if all(remaining_is_init):
            y_query_preds, _ = self.field(
                pos_query=pos_query,
                edge_index_query=[],
                pos_compose=compose_pos,
                node_attr_compose=h_compose,
                edge_index_q_cps_knn=query_compose_knn_edge_index,
            )
        else:
            assert not any(
                remaining_is_init
            )  # Due to the assumption `all(is_init) or not any(is_init)`
            edge_index_query_i = []
            for i, size in enumerate(remaining_ligand_context_sizes):
                edge_index_query_i.extend([i] * size)
            edge_index_query_i = torch.tensor(edge_index_query_i, device=device)
            assert len(edge_index_query_i) == len(idx_ligand)
            edge_index_query_j = added_concat(
                [
                    torch.arange(size, device=device)
                    for size in remaining_ligand_context_sizes
                ],
                remaining_compose_sizes,
                dim=0,
            )
            edge_index_query = torch.stack(
                [edge_index_query_i, edge_index_query_j], dim=0
            )

            (
                index_real_cps_edge_for_atten,
                tri_edge_index,
                tri_edge_feat,
            ) = self._get_tri_edges(
                edge_index_query=edge_index_query,
                pos_query=pos_query,
                idx_ligand=idx_ligand,
                ligand_bond_index=ligand_context_bond_index,
                ligand_bond_type=ligand_context_bond_type,
                modified=True,
            )
            y_query_preds, edge_preds = self.field(
                pos_query=pos_query,
                edge_index_query=edge_index_query,
                pos_compose=compose_pos,
                node_attr_compose=h_compose,
                edge_index_q_cps_knn=query_compose_knn_edge_index,
                index_real_cps_edge_for_atten=index_real_cps_edge_for_atten,
                tri_edge_index=tri_edge_index,
                tri_edge_feat=tri_edge_feat,
            )

            """
            raise Exception(
                pos_query,
                query_compose_knn_edge_index,
                index_real_cps_edge_for_atten,
                tri_edge_index,
                tri_edge_feat,
                y_query_preds,
                edge_preds,
            )
            """

        assert len(y_query_preds.shape) == 2 and y_query_preds.shape[0] == len(
            remaining_idxs
        )
        if not all(remaining_is_init):
            assert edge_preds.shape == (len(idx_ligand), self.num_edge_types), (
                edge_preds.shape,
                (len(idx_ligand), self.num_edge_types),
            )
        for i, idx in enumerate(remaining_idxs):
            y_query_pred = y_query_preds[i]
            pi_element = ElementDistribution(y_query_pred)
            pi_element_list[idx] = pi_element

            element = pi_element.sample(seed=seed, generator=generator)
            element_list[idx] = element

            if return_activation_stat:
                element_logits = (
                    y_query_pred.cpu().numpy().astype(np.float64)
                )  # for numerical stability
                act_stats[idx]["element_logit_mean"] = np.mean(element_logits)
                act_stats[idx]["element_logit_exp_sum_log"] = np.log(
                    np.exp(element_logits).sum()
                )  # used to compute has_atom threshold
                act_stats[idx]["element_prob_sampled"] = np.exp(
                    pi_element.get_logp(element).item()
                )

            if not all(remaining_is_init):
                edge_pred = decompose_tensors_and_getindex(
                    edge_preds, remaining_ligand_context_sizes, i, dim=0
                )
                pi_bond = BondDistribution(edge_pred)
                pi_bond_list[idx] = pi_bond

                bond = pi_bond.sample(seed=seed, generator=generator)
                bond_list[idx] = bond

                if return_activation_stat:
                    _, bond_types = bond
                    nonzero_bond_idxs = torch.where(bond_types > 0)[0]
                    nonzero_bond_types = bond_types[nonzero_bond_idxs]
                    assert torch.all(nonzero_bond_types > 0)
                    if len(nonzero_bond_idxs) > 0:
                        act_stats[idx]["has_bond"] = True
                        act_stats[idx]["bond_prob_n"] = len(nonzero_bond_idxs)
                        nonzero_bond_probs = pi_bond.probs[
                            nonzero_bond_idxs, nonzero_bond_types
                        ]
                        assert nonzero_bond_probs.shape == (len(nonzero_bond_idxs),)
                        act_stats[idx][
                            "bond_prob_sum"
                        ] = nonzero_bond_probs.sum().item()
                        act_stats[idx]["bond_prob_sq_sum"] = (
                            (nonzero_bond_probs**2).sum().item()
                        )
                    else:
                        act_stats[idx]["has_bond"] = False

            else:
                if return_activation_stat:
                    act_stats[idx]["has_bond"] = False

        # Wrap up the results
        for i, idx in enumerate(remaining_idxs):
            assert idx_focal_list[idx] is not None
            assert pos_list[idx] is not None
            assert element_list[idx] is not None
            if remaining_is_init[i]:
                assert bond_list[idx] is None
                bond_index, bond_type = None, None
            else:
                assert bond_list[idx] is not None
                bond_index, bond_type = bond_list[idx]
            action = AtomAction(
                idx_focal_list[idx],
                pos_list[idx],
                element_list[idx],
                bond_index,
                bond_type,
            )
            actions[idx] = action

            assert pi_idx_focal_list[idx] is not None
            assert pi_pos_list[idx] is not None
            assert pi_element_list[idx] is not None
            if remaining_is_init[i]:
                assert pi_bond_list[idx] is None
            else:
                assert pi_bond_list[idx] is not None

            pi = ActionConditionalDistribution(
                condition=actions[idx],
                pi_idx_focal=pi_idx_focal_list[idx],
                pi_pos=pi_pos_list[idx],
                pi_element=pi_element_list[idx],
                pi_bond=pi_bond_list[idx],
            )
            pis[idx] = pi

            logp = pi.get_logp(actions[idx])
            logps[idx] = logp

        assert all(action is not None for action in actions)
        assert all(logp is not None for logp in logps)
        assert all(pi is not None for pi in pis)

        for i, action in enumerate(actions):
            actions[i] = action.to(data_device)

        if return_activation_stat:
            return actions, (logps, pis), act_stats
        else:
            return actions, (logps, pis)

    def sample_one(
        self,
        data=None,
        pos_only_mean=False,
        hard_stop=None,
        compose_feature=None,
        compose_pos=None,
        idx_ligand=None,
        idx_protein=None,
        compose_knn_edge_index=None,
        compose_knn_edge_feature=None,
        ligand_context_bond_index=None,
        ligand_context_bond_type=None,
        seed=None,
        generator=None,
    ) -> Tuple[Action, Tuple[Tensor, ActionConditionalDistribution]]:
        """Sample one "action"(properties of a new atom), along with the log probabilities
        and conditional distributions of the each component of the action"""
        if data is not None:
            data_device = data.compose_pos.device
            device = next(self.parameters()).device
            batch = Batch.from_data_list([data]).to(device)
            compose_feature = batch.compose_feature.float()
            compose_pos = batch.compose_pos
            idx_ligand = batch.idx_ligand_ctx_in_compose
            idx_protein = batch.idx_protein_in_compose
            compose_knn_edge_index = batch.compose_knn_edge_index
            compose_knn_edge_feature = batch.compose_knn_edge_feature
            ligand_context_bond_index = batch.ligand_context_bond_index
            ligand_context_bond_type = batch.ligand_context_bond_type
        else:
            assert compose_feature is not None
            assert compose_pos is not None
            assert idx_ligand is not None
            assert idx_protein is not None
            assert compose_knn_edge_index is not None
            assert compose_knn_edge_feature is not None
            assert ligand_context_bond_index is not None
            assert ligand_context_bond_type is not None

        is_init = len(idx_ligand) == 0

        if is_init:
            pi_idx_focal, h_compose = self._compute_init_pi_idx_focal(
                compose_feature,
                compose_pos,
                idx_protein,
                compose_knn_edge_index,
                compose_knn_edge_feature,
            )
        else:
            if hasattr(data, "ligand_focalizable_mask"):
                focalizable_mask = data.ligand_focalizable_mask.to(device)
            else:
                focalizable_mask = None
            pi_idx_focal, h_compose = self._compute_pi_idx_focal(
                compose_feature,
                compose_pos,
                idx_ligand,
                idx_protein,
                compose_knn_edge_index,
                compose_knn_edge_feature,
                focalizable_mask=focalizable_mask,
            )
        idx_focal = pi_idx_focal.sample(
            hard_stop=hard_stop, seed=seed, generator=generator
        )

        if idx_focal is None:
            assert not is_init

            action = FocalStopAction()
            pi = ActionConditionalDistribution(
                condition=action,
                pi_idx_focal=pi_idx_focal,
                pi_pos=None,
                pi_element=None,
                pi_bond=None,
            )

        else:  # focal sampled
            pi_pos = self._compute_pi_pos(h_compose, compose_pos, idx_focal)
            pos = pi_pos.sample(only_mean=pos_only_mean, seed=seed, generator=generator)

            if is_init:
                pi_element = self._compute_init_pi_element(
                    pos,
                    h_compose,
                    compose_pos,
                )
                element = pi_element.sample(seed=seed, generator=generator)
                pi_bond, bond_index, bond_type = None, None, None
            else:
                pi_element, pi_bond = self._compute_pi_element_bond(
                    pos,
                    h_compose,
                    compose_pos,
                    idx_ligand,
                    ligand_context_bond_index,
                    ligand_context_bond_type,
                )
                element = pi_element.sample(seed=seed, generator=generator)
                bond_index, bond_type = pi_bond.sample(seed=seed, generator=generator)

            action = AtomAction(
                idx_focal=idx_focal,
                pos=pos,
                element=element,
                bond_index=bond_index,
                bond_type=bond_type,
            )
            pi = ActionConditionalDistribution(
                condition=action,
                pi_idx_focal=pi_idx_focal,
                pi_pos=pi_pos,
                pi_element=pi_element,
                pi_bond=pi_bond,
            )

        logp = pi.get_logp(action)

        if data is not None:
            action = action.to(data_device)

        return (action, (logp, pi))

    def _compute_init_pi_idx_focal(
        self,
        compose_feature,
        compose_pos,
        idx_protein,
        compose_knn_edge_index,
        compose_knn_edge_feature,
    ) -> Tuple[IdxFocalDistribution, Tensor]:
        """

        return:
            pi_idx_focal, h_compose
        """
        # # 0: encode
        h_compose = self._embed_compose(
            compose_feature,
            compose_pos,
            idx_ligand=torch.empty(0).to(idx_protein),
            idx_protein=idx_protein,
            compose_knn_edge_index=compose_knn_edge_index,
            compose_knn_edge_feature=compose_knn_edge_feature,
        )

        # # 1: predict frontier
        y_frontier_pred = self.frontier_pred(
            h_compose,
            idx_protein,
        )[:, 0]
        pi_idx_focal = IdxFocalDistribution(
            y_frontier_pred, allow_stop=False, stop_first=self.focal_action_stop_first
        )

        return (pi_idx_focal, h_compose)

    def _compute_pi_idx_focal(
        self,
        compose_feature,
        compose_pos,
        idx_ligand,
        idx_protein,
        compose_knn_edge_index,
        compose_knn_edge_feature,
        focalizable_mask=None,
    ) -> Tuple[IdxFocalDistribution, Tensor]:
        """

        return:
            pi_idx_focal, h_compose
        """
        # # 0: encode
        h_compose = self._embed_compose(
            compose_feature,
            compose_pos,
            idx_ligand,
            idx_protein,
            compose_knn_edge_index,
            compose_knn_edge_feature,
        )

        # # 1: predict frontier
        y_frontier_pred = self.frontier_pred(
            h_compose,
            idx_ligand,
        )[:, 0]
        pi_idx_focal = IdxFocalDistribution(
            y_frontier_pred,
            stop_first=self.focal_action_stop_first,
            focalizable_mask=focalizable_mask,
        )

        return (pi_idx_focal, h_compose)

    def _compute_pi_pos(
        self, h_compose, compose_pos, idx_focal_in_compose
    ) -> PosDistribution:
        """
        Returns:
            pi_pos (PosDistribution)
        """
        relative_pos_mu, abs_pos_mu, pos_sigma, pos_pi = self.pos_predictor(
            h_compose,
            idx_focal_in_compose.unsqueeze(0),
            compose_pos,
        )
        abs_pos_mu = abs_pos_mu.squeeze(0)
        pos_sigma = pos_sigma.squeeze(0)
        pos_pi = pos_pi.squeeze(0)

        pi_pos = PosDistribution(abs_pos_mu, pos_sigma, pos_pi)
        return pi_pos

    def _compute_init_pi_element(
        self,
        pos,
        h_compose,
        compose_pos,
    ):
        query_compose_knn_edge_index = knn(
            x=compose_pos, y=pos.unsqueeze(0), k=self.config.field.knn, num_workers=16
        )
        y_query_pred, _ = self.field(
            pos_query=pos.unsqueeze(0),
            edge_index_query=[],
            pos_compose=compose_pos,
            node_attr_compose=h_compose,
            edge_index_q_cps_knn=query_compose_knn_edge_index,
        )
        y_query_pred = y_query_pred.squeeze(0)

        pi_element = ElementDistribution(y_query_pred)
        return pi_element

    def _compute_pi_element_bond(
        self,
        pos,
        h_compose,
        compose_pos,
        idx_ligand,
        ligand_bond_index,
        ligand_bond_type,
    ) -> Tuple[ElementDistribution, BondDistribution]:
        """
        Returns:
            pi_element (ElementDistribution)
            pi_bond (BondDistribution)
        """
        y_query_pred, edge_pred = self._query_position(
            pos_query=pos.unsqueeze(0),
            h_compose=h_compose,
            compose_pos=compose_pos,
            idx_ligand=idx_ligand,
            ligand_bond_index=ligand_bond_index,
            ligand_bond_type=ligand_bond_type,
        )
        y_query_pred = y_query_pred.squeeze(0)
        edge_pred = edge_pred.squeeze(0)

        pi_element = ElementDistribution(y_query_pred)
        pi_bond = BondDistribution(edge_pred)
        return (pi_element, pi_bond)
