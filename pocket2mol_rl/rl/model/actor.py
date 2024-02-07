import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from rdkit.Chem.rdchem import Mol as RdkitMol

from pocket2mol_rl.data.action import (
    ActionConditionalDistribution,
    AtomAction,
    FocalStopAction,
    StopAction,
)
from pocket2mol_rl.data.add_atom import get_data_with_action_applied, get_past_data
from pocket2mol_rl.data.data import ProteinLigandData
from pocket2mol_rl.data.transform.composer import AtomComposer
from pocket2mol_rl.data.transform.featurize import (
    FeaturizeLigandAtom,
    FeaturizeProteinAtom,
)
from pocket2mol_rl.models.maskfill import MaskFillModelVN
from pocket2mol_rl.rl.component.episode import (
    Episode,
    FailedCompleteEpisode,
    FailedIntermediateEpisode,
    SuccessfulEpisode,
)
from pocket2mol_rl.treemeans.data import PlanarTree
from pocket2mol_rl.treemeans.data.transform.featurize import (
    ToyFeaturizeLigandAtom,
    ToyFeaturizeProteinAtom,
)
from pocket2mol_rl.treemeans.evaluation.data_structure import (
    SimplePlanarTree,
    TreeReconsError,
)
from pocket2mol_rl.treemeans.models.maskfill import ToyMaskFillModelVN
from pocket2mol_rl.utils.batch_computation import group_compute_merge
from pocket2mol_rl.utils.reconstruct import (
    MolReconsError,
    reconstruct_from_generated_with_edges,
)


class Actor:
    def __init__(
        self,
        model: MaskFillModelVN,
        abort_middle=False,
        focal_action_stop_first=False,
        generation_max_length: int = 128,
    ):
        """
        Args:
            model (nn.Module): The language model used to generate new samples.
                Model is required to implement .generate and .forward methods.

        """
        self.model = model
        self.abort_middle = abort_middle
        self.focal_action_stop_first = focal_action_stop_first
        self.model.set_focal_action_stop_first(focal_action_stop_first)
        self.generation_max_length = generation_max_length

        self.featurize_module = type(self).get_ligand_featurizer()
        self.atom_composer = self.get_atom_composer()

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, generation_max_length=None, device="cuda"):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        d = torch.load(ckpt_path, map_location=device)
        if "rl_config" in d:
            actor_config = d["rl_config"]["rl"]["actor"]
        else:
            actor_config = {"abort_middle": True, "max_length": 64}
        abort_middle = actor_config.get("abort_middle", False)
        focal_action_stop_first = actor_config.get("focal_action_stop_first", False)
        if generation_max_length is None:
            generation_max_length = actor_config.get("max_length", 128)

        protein_featurizer = cls.get_protein_featurizer()
        ligand_featurizer = cls.get_ligand_featurizer()

        model = MaskFillModelVN(
            d["config"].model,
            ligand_featurizer=ligand_featurizer,
            protein_featurizer=protein_featurizer,
        ).to(device)
        model.load_state_dict(d["model"])

        return cls(
            model,
            abort_middle=abort_middle,
            focal_action_stop_first=focal_action_stop_first,
            generation_max_length=generation_max_length,
        )

        # TODO: here

    def to(self, device):
        self.model = self.model.to(device)

    @property
    def device(self):
        return self.model.device

    @classmethod
    def get_ligand_featurizer(cls):
        raise NotImplementedError()

    @classmethod
    def get_protein_featurizer(cls):
        raise NotImplementedError()

    def get_atom_composer(self):  # requires self.model
        raise NotImplementedError()

    def parse_final_data(
        self, data: ProteinLigandData
    ) -> Tuple[Optional[Any], Optional[str]]:
        raise NotImplementedError()

    def _is_valid_intermediate(
        self, data: ProteinLigandData, action: AtomAction
    ) -> bool:
        raise NotImplementedError()

    @torch.no_grad()
    def get_rollouts_in_batch(
        self,
        data_batch: List[ProteinLigandData],
        pos_only_mean=False,
        hard_stop=None,
        seed=None,
        generator=None,
        replace_conditions=None,
        return_activation_stat=False,
    ) -> List[Episode]:
        """
        Args:
            replace_conditions (Optional[List[List[Action]]]): For debugging, replace the actions that go into `get_data_with_action_applied` with pre-computed actions. In this case, the `actions` attributes in the returned episodes will still be those from the model. Namely, only the rollout path is replaced. This is useful for debugging, since without this, two rollouts can deviate from each other significantly even if each step differs slightly.
        """
        if replace_conditions is not None:
            assert len(replace_conditions) == len(data_batch)

        # It is okay that the data in data_batch has data.ligand_context_size > 0
        """ 
        assert all(data.ligand_context_size == 0 for data in data_batch), [
            data.ligand_context_size for data in data_batch
        ]
        """
        if hard_stop is not None:
            assert self.focal_action_stop_first

        actions_batch = [[] for _ in range(len(data_batch))]
        logp_batch = [[] for _ in range(len(data_batch))]
        pis_batch = [[] for _ in range(len(data_batch))]
        episodes = [None for _ in range(len(data_batch))]

        remaining_idxs = list(range(len(data_batch)))

        if return_activation_stat:
            act_stat = defaultdict(float)  # This will be used tom compute act_stat

        for step in range(self.generation_max_length):
            if len(remaining_idxs) == 0:
                break
            remaining_data_batch = [data_batch[idx] for idx in remaining_idxs]

            if return_activation_stat:
                (
                    action_batch,
                    (_logp_batch, pi_batch),
                    step_act_stats,
                ) = self.model.sample_one_in_batch(
                    remaining_data_batch,
                    pos_only_mean=pos_only_mean,
                    hard_stop=hard_stop,
                    seed=seed,
                    generator=generator,
                    return_activation_stat=True,
                )

                for step_act_stat in step_act_stats:
                    if step_act_stat["focal_stop"]:
                        continue
                    act_stat["focal_logit_n"] += step_act_stat["focal_logit_n"]
                    act_stat["focal_logit_sum"] += step_act_stat["focal_logit_sum"]
                    act_stat["focal_logit_sq_sum"] += step_act_stat[
                        "focal_logit_sq_sum"
                    ]

                    act_stat["mu_pdf_mean_n"] += 1
                    act_stat["mu_pdf_mean_sum"] += step_act_stat["mu_pdf_mean"]
                    act_stat["mu_pdf_mean_sq_sum"] += step_act_stat["mu_pdf_mean"] ** 2
                    act_stat["top-5_mu_pdf_mean_sum"] += step_act_stat[
                        "top-5_mu_pdf_mean"
                    ]
                    act_stat["top-5_mu_pdf_mean_sq_sum"] += (
                        step_act_stat["top-5_mu_pdf_mean"] ** 2
                    )

                    act_stat["element_logit_n"] += 1
                    act_stat["element_logit_mean_sum"] += step_act_stat[
                        "element_logit_mean"
                    ]
                    act_stat["element_logit_mean_sq_sum"] += (
                        step_act_stat["element_logit_mean"] ** 2
                    )
                    act_stat["element_logit_exp_sum_log_sum"] += step_act_stat[
                        "element_logit_exp_sum_log"
                    ]
                    act_stat["element_logit_exp_sum_log_sq_sum"] += (
                        step_act_stat["element_logit_exp_sum_log"] ** 2
                    )
                    act_stat["element_prob_sampled_sum"] += step_act_stat[
                        "element_prob_sampled"
                    ]
                    act_stat["element_prob_sampled_sq_sum"] += (
                        step_act_stat["element_prob_sampled"] ** 2
                    )

                    if not step_act_stat["has_bond"]:
                        continue
                    act_stat["bond_prob_n"] += step_act_stat["bond_prob_n"]
                    act_stat["bond_prob_sum"] += step_act_stat["bond_prob_sum"]
                    act_stat["bond_prob_sq_sum"] += step_act_stat["bond_prob_sq_sum"]

            else:
                action_batch, (_logp_batch, pi_batch) = self.model.sample_one_in_batch(
                    remaining_data_batch,
                    pos_only_mean=pos_only_mean,
                    hard_stop=hard_stop,
                    seed=seed,
                    generator=generator,
                )
            assert len(action_batch) == len(remaining_idxs)
            assert len(_logp_batch) == len(remaining_idxs)
            assert len(pi_batch) == len(remaining_idxs)

            finished_idxs = []
            for i, idx in enumerate(remaining_idxs):
                _action = action_batch[i]
                if replace_conditions is None:
                    action = _action
                else:
                    assert step < len(replace_conditions[idx])
                    action = replace_conditions[idx][step]

                actions_batch[idx].append(_action)
                logp_batch[idx].append(_logp_batch[i])
                pis_batch[idx].append(pi_batch[i])

                if isinstance(action, StopAction):
                    assert isinstance(action, FocalStopAction)
                    obj, msg = self.parse_final_data(data_batch[idx])
                    logp = torch.stack(logp_batch[idx], dim=0)
                    if obj is None:
                        episode = FailedCompleteEpisode(
                            data_batch[idx],
                            actions_batch[idx],
                            logp,
                            pis_batch[idx],
                        )
                    else:
                        episode = SuccessfulEpisode(
                            obj,
                            data_batch[idx],
                            actions_batch[idx],
                            logp,
                            pis_batch[idx],
                        )
                    episodes[idx] = episode
                    finished_idxs.append(idx)
                    continue
                else:
                    assert isinstance(action, AtomAction)

                    data_batch[idx] = get_data_with_action_applied(
                        data_batch[idx], action, self.featurize_module
                    )
                    data_batch[idx] = self.atom_composer(data_batch[idx])
                    ok, msg = self._is_valid_intermediate(data_batch[idx], action)
                    if not ok:
                        logp = torch.stack(logp_batch[idx], dim=0)
                        episode = FailedIntermediateEpisode(
                            data_batch[idx],
                            actions_batch[idx],
                            logp,
                            pis_batch[idx],
                            failure_msg=msg,
                        )
                        episodes[idx] = episode
                        finished_idxs.append(idx)
                        continue

            for idx in finished_idxs:
                remaining_idxs.remove(idx)

        remaining_idxs = set(remaining_idxs)

        for i, episode in enumerate(episodes):
            if i in remaining_idxs:
                assert episode is None
                data = data_batch[i]
                actions = actions_batch[i]
                logp = torch.stack(logp_batch[i], dim=0)
                pis = pis_batch[i]
                episodes[i] = FailedIntermediateEpisode(data, actions, logp, pis)
            else:
                assert episode is not None

        assert all(isinstance(episode, Episode) for episode in episodes)
        if return_activation_stat:
            act_stat["top-5_mu_pdf_mean_n"] = act_stat["mu_pdf_mean_n"]
            act_stat["element_logit_mean_n"] = act_stat["element_logit_n"]
            act_stat["element_logit_exp_sum_log_n"] = act_stat["element_logit_n"]
            act_stat["element_prob_sampled_n"] = act_stat["element_logit_n"]

            return episodes, act_stat
        else:
            return episodes

    @torch.no_grad()
    def get_rollout(
        self,
        data: ProteinLigandData,
        pos_only_mean=False,
        hard_stop=None,
        seed=None,
        generator=None,
    ) -> Episode:
        """Get a rollout from base data to the end of the episode

        Args:
            data (ProteinLigandData): base data

        Returns:
            Episode: Resulting episode
        """
        if hard_stop is not None:
            assert self.focal_action_stop_first
        assert data.ligand_context_size == 0, data.ligand_context_size
        actions = []
        logp = []
        pis = []
        for _ in range(self.generation_max_length):
            action, (_logp, pi) = self.model.sample_one(
                data=data,
                pos_only_mean=pos_only_mean,
                hard_stop=hard_stop,
                seed=seed,
                generator=generator,
            )
            actions.append(action)
            logp.append(_logp)
            pis.append(pi)

            if isinstance(action, StopAction):
                assert isinstance(action, FocalStopAction)
                obj, msg = self.parse_final_data(data)
                logp = torch.stack(logp, dim=0)
                if obj is None:
                    # TODO: log the failure message?
                    return FailedCompleteEpisode(data, actions, logp, pis)
                else:
                    return SuccessfulEpisode(obj, data, actions, logp, pis)
            else:
                assert isinstance(action, AtomAction)

                data = get_data_with_action_applied(data, action, self.featurize_module)
                data = self.atom_composer(data)
                ok, msg = self._is_valid_intermediate(data, action)
                if not ok:
                    logp = torch.stack(logp, dim=0)
                    return FailedIntermediateEpisode(
                        data, actions, logp, pis, failure_msg=msg
                    )

        logp = torch.stack(logp, dim=0)
        return FailedIntermediateEpisode(data, actions, logp, pis)

    def get_rollouts(
        self,
        data_batch: Union[List[ProteinLigandData], ProteinLigandData],
        compute_method="batched",
        pos_only_mean=False,
        hard_stop=None,
        seed: Optional[int] = None,
        generator=None,
        replace_conditions=None,
        bs_width=None,
        bs_num_samples_per_data=None,
        bs_max_num_samples=None,
        bs_max_num_steps=None,
        only_successful=False,
    ) -> List[Episode]:
        """
        Args:
            data_batch (List[ProteinLigandData]):
            seed (Optional[int]): Random seed for sampling actions, used in torch.Generator.manual_seed(seed) every time before sampling an action. Used for debugging.
            replace_conditions (Optional[List[Action]]): For debugging, replace the actions that go into `get_data_with_action_applied` with pre-computed actions. In this case, the `actions` attributes in the returned episodes will still be those from the model. Namely, only the rollout path is replaced. This is useful for debugging, since without this, two rollouts can deviate from each other significantly even if each step differs slightly.

        Returns:
            episodes (List[Episode])
        """

        if compute_method == "batched":
            assert isinstance(data_batch, list)
            if only_successful:
                raise NotImplementedError()
            return self.get_rollouts_in_batch(
                data_batch,
                pos_only_mean=pos_only_mean,
                hard_stop=hard_stop,
                seed=seed,
                generator=generator,
                replace_conditions=replace_conditions,
            )
        elif compute_method == "iterative":
            assert isinstance(data_batch, list)
            assert replace_conditions is None
            if only_successful:
                raise NotImplementedError()
            episodes = []
            for data in data_batch:
                episode = self.get_rollout(
                    data,
                    pos_only_mean=pos_only_mean,
                    hard_stop=hard_stop,
                    seed=seed,
                    generator=generator,
                )
                episodes.append(episode)
            return episodes
        else:
            raise ValueError(f"Invalid compute_method {compute_method}")

    def get_logit_in_batch(
        self, episodes: List[Episode]
    ) -> Tuple[List[torch.Tensor], List[List[ActionConditionalDistribution]]]:
        sizes = np.array(
            [episode.final_data.ligand_context_size for episode in episodes]
        )
        num_states = np.array(
            [
                (
                    sizes[i]
                    if isinstance(episodes[i], FailedIntermediateEpisode)
                    else sizes[i] + 1
                )
                for i in range(len(episodes))
            ]
        )
        assert all(
            num_states[i] == len(episodes[i].actions) for i in range(len(episodes))
        )
        logps = [[] for _ in range(len(episodes))]
        pis = [[] for _ in range(len(episodes))]

        for i in range(np.max(num_states)):
            idxs = np.where(num_states > i)[0]
            assert len(idxs) > 0
            data_list = [
                self.atom_composer(
                    get_past_data(episodes[idx].final_data, i, self.featurize_module)
                )
                for idx in idxs
            ]
            actions = [episodes[idx].actions[i] for idx in idxs]
            _logps, _pis = self.model.compute_probs_off_policy_in_batch(
                data_list, actions
            )
            for j, idx in enumerate(idxs):
                logps[idx].append(_logps[j])
                pis[idx].append(_pis[j])

        logps = [torch.stack(logp, dim=0) for logp in logps]
        assert all(len(logp.shape) == 1 for logp in logps)
        return (logps, pis)

    def get_logit(
        self, episode: Episode
    ) -> Tuple[torch.Tensor, List[ActionConditionalDistribution]]:
        n = episode.final_data.ligand_context_size  # TODO: check the number
        logp = []
        pis = []
        for i in range(n + 1):
            if i < n:
                data = self.atom_composer(
                    get_past_data(episode.final_data, i, self.featurize_module)
                )
                action = episode.actions[i]
            else:
                if isinstance(episode, FailedIntermediateEpisode):
                    assert len(episode) == n
                    break
                assert isinstance(episode, SuccessfulEpisode) or isinstance(
                    episode, FailedCompleteEpisode
                )
                assert len(episode) == n + 1
                assert isinstance(episode.actions[-1], StopAction)
                data = episode.final_data
                action = episode.actions[-1]
            _logp, pi = self.model.compute_probs_off_policy(data=data, action=action)
            logp.append(_logp)
            pis.append(pi)
        logp = torch.stack(logp, dim=0)
        assert len(logp.shape) == 1
        return (logp, pis)

    def get_logits(
        self,
        episodes: List[Episode],
        state_idxs: Optional[List[int]] = None,
        compute_method="batched",
    ) -> Union[
        Tuple[List[torch.Tensor], List[List[ActionConditionalDistribution]]],
        Tuple[List[torch.Tensor], List[ActionConditionalDistribution]],
    ]:
        """
        Inputs:
            episodes (List[Episode])
            state_idxs (Optional[List[int]]): If not None, only compute the logit for episodes[i] at the state_idxs[i]-th state. This is useful for RL training with a replay buffer.
        Outputs:
            logp_batch (List[Tensor]): logp_batch[i][j] is the logp of the j-th action in the i-th episode
            pis_batch (List[List[ActionConditionalDistribution]]): pis_batch[i][j] is the pi of the j-th action in the i-th episode
        """
        if state_idxs is None:
            if compute_method == "batched":
                return self.get_logit_in_batch(episodes)
            elif compute_method == "iterative":
                logp_batch = []
                pis_batch = []
                for episode in episodes:
                    logp, pis = self.get_logit(episode)
                    logp_batch.append(logp)
                    pis_batch.append(pis)

                return (logp_batch, pis_batch)
            else:
                raise ValueError(f"Invalid compute_method {compute_method}")
        else:
            if compute_method != "batched":
                raise NotImplementedError()
            assert len(episodes) == len(state_idxs)
            assert all(0 <= i < len(e) for e, i in zip(episodes, state_idxs))

            data_list = [
                self.atom_composer(
                    get_past_data(e.final_data, i, self.featurize_module)
                )
                for e, i in zip(episodes, state_idxs)
            ]
            action_list = [e.actions[i] for e, i in zip(episodes, state_idxs)]

            # compute for inits and non-inits separately
            # TODO: inits and non-inits simultaneously?
            logps, pis = group_compute_merge(
                lambda data, action: data.ligand_context_size == 0,
                lambda _data_list, _action_list: self.model.compute_probs_off_policy_in_batch(
                    _data_list, _action_list
                ),
                data_list,
                action_list,
            )
            assert len(logps) == len(pis) == len(episodes)
            return (logps, pis)

    def freeze(
        self,
        config: argparse.Namespace,
    ):
        """Freeze the encoder, embbeding of model
        Also freeze the field network except for the last layer (edge_pred and classifier)
        """
        # Freeze parameters according to config
        freeze_emb = config.get("freeze_emb", True)
        freeze_encoder = config.get("freeze_encoder", True)
        freeze_field = config.get("freeze_field", True)
        freeze_mu = config.get("freeze_mu", False)

        if freeze_emb:
            for param in self.model.ligand_atom_emb.parameters():
                param.requires_grad = False

            for param in self.model.protein_atom_emb.parameters():
                param.requires_grad = False

        if freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        if freeze_field:
            # For field network
            for param in self.model.field.parameters():
                param.requires_grad = False

            for param in self.model.field.edge_pred.parameters():
                param.requires_grad = True
            for param in self.model.field.classifier.parameters():
                param.requires_grad = True

        if freeze_mu:
            self.model.pos_predictor.freeze_mu()


class MolgenActor(Actor):
    def __init__(self, model: ToyMaskFillModelVN, **kwargs):
        super().__init__(model, **kwargs)

    @classmethod
    def get_ligand_featurizer(cls):
        return FeaturizeLigandAtom()

    @classmethod
    def get_protein_featurizer(cls):
        return FeaturizeProteinAtom()

    def get_atom_composer(self):
        return AtomComposer(
            self.model.config.encoder.knn, self.model.config.encoder.num_edge_types
        )

    def parse_final_data(
        self, data: ProteinLigandData
    ) -> Tuple[Optional[RdkitMol], Optional[str]]:
        try:
            rd_mol = reconstruct_from_generated_with_edges(
                data, raise_error=True, sanitize=True
            )
        except MolReconsError as e:
            return (None, str(e))
        except Exception as e:
            raise e

        return (rd_mol, None)

    def _is_valid_intermediate(self, data: ProteinLigandData, action: AtomAction):
        if not self.abort_middle:
            return True, None
        if data.ligand_context_size <= 1:
            return True, None

        # action contains at least one bond
        try:
            num_new_bonds = len(torch.where(action.bond_type > 0)[0])
        except:
            print(type(action))
            print(action.idx_focal)
            print(action.element)
            print(action.bond_index)
            print(action.bond_type)
            raise Exception()

        if num_new_bonds == 0:
            return False, "0 new bonds"

        def _get_max_possible_valences():
            l = []
            for atomic_num in data.ligand_context_element.tolist():
                v = self.featurize_module.max_possible_valences[atomic_num]
                l.append(v)
            return l

        def _get_valences(n):
            bond_degrees = {}
            counted = {}

            m = len(data.ligand_context_bond_type)
            assert data.ligand_context_bond_type.shape == (m,)
            assert data.ligand_context_bond_index.shape == (2, m)

            for i in range(m):
                a = data.ligand_context_bond_index[0, i]
                b = data.ligand_context_bond_index[1, i]
                t = data.ligand_context_bond_type[i]
                assert 0 <= a < n
                assert 0 <= b < n

                if not (a in counted and b in counted[a]):
                    counted.setdefault(a, set([])).add(b)
                    bond_degrees.setdefault(a, []).append(t)
                if not (b in counted and a in counted[b]):
                    counted.setdefault(b, set([])).add(a)
                    bond_degrees.setdefault(b, []).append(t)

            valences = [0 for _ in range(n)]
            for i in bond_degrees:
                valences[i] = sum(bond_degrees[i])

            return valences

        # check if each atom does not exceed maximum non-hydrogen valences
        max_possible_valences = _get_max_possible_valences()
        valences = _get_valences(len(max_possible_valences))
        assert len(valences) == len(max_possible_valences)
        if any(v > m for v, m in zip(valences, max_possible_valences)):
            return False, "Exceed maximum non-hydrogen valences"

        if hasattr(data, "ligand_branch_point_mask"):
            # Ensure that all bonds are originated from the same branch point as the focal

            assert hasattr(data, "ligand_focalizable_mask")
            ligand_branch_point_mask = data.ligand_branch_point_mask
            ligand_focalizable_mask = data.ligand_focalizable_mask
            assert ligand_branch_point_mask.shape == (data.ligand_context_size,)
            focal_idxs = data.idx_focal

            assert ligand_focalizable_mask[action.idx_focal]

            # Check whether the focal atom has any connection with a focalizable atom originated from another branch

            roots = torch.tensor(
                [-1 for _ in range(data.ligand_context_size)], device=focal_idxs.device
            )

            idx_generated = -1
            for i in range(data.ligand_context_size):
                if not ligand_focalizable_mask[i]:
                    continue
                if ligand_branch_point_mask[i]:
                    roots[i] = i
                else:
                    idx_generated += 1
                    focal_idx = int(focal_idxs[idx_generated].item())
                    assert ligand_focalizable_mask[focal_idx]
                    roots[i] = roots[focal_idx]
                assert ligand_branch_point_mask[roots[i]]

            assert torch.all(action.bond_type > 0)
            for idx in action.bond_index[1, :].tolist():
                if roots[idx] != roots[action.idx_focal.item()]:
                    return (
                        False,
                        "Connect to a focalizable atom originated from another branch",
                    )

        return True, None


class TreemeansActor(Actor):
    def __init__(self, model: ToyMaskFillModelVN, **kwargs):
        super().__init__(model, **kwargs)

    @classmethod
    def get_ligand_featurizer(cls):
        return ToyFeaturizeLigandAtom()

    @classmethod
    def get_protein_featurizer(cls):
        return ToyFeaturizeProteinAtom()

    def get_atom_composer(self):
        return AtomComposer(
            self.model.config.encoder.knn, self.model.config.encoder.num_edge_types
        )

    def parse_final_data(
        self, data: PlanarTree
    ) -> Tuple[Optional[PlanarTree], Optional[str]]:
        try:
            tree = SimplePlanarTree.init_from_data(data, fill_context=True)
        except TreeReconsError as e:
            return (None, str(e))
        except Exception as e:
            raise e

        return (tree, None)

    def _is_valid_action(self, action: AtomAction):
        """Check the action has only one bond"""
        assert action.bond_index.shape[1] == action.bond_type.shape[0]
        num = action.bond_index.shape[1]
        if num == 1:
            return True, None
        else:
            return False, f"The action has {num} bonds"

    def _is_valid_intermediate(self, data: PlanarTree, action: AtomAction):
        if not self.abort_middle:
            return True, None
        if data.ligand_context_size <= 1:
            return True, None
        # Check if the generated data is tree
        ok, msg = self._is_valid_action(action)
        if not ok:
            return False, msg
        try:
            tree = SimplePlanarTree.init_from_data(data, fill_context=True)
        except TreeReconsError:
            return False, "The generated data is not tree"
        except Exception as e:
            raise e
        return True, None
