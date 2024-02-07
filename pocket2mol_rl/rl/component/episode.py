from abc import ABCMeta
from copy import deepcopy
from typing import List

import torch
from rdkit.Chem import Mol as RdkitMol
from torch import Tensor

from pocket2mol_rl.data.action import (
    Action,
    ActionConditionalDistribution,
    StopAction,
)
from pocket2mol_rl.data.add_atom import get_data_with_action_applied


class Episode:
    def __init__(
        self,
        final_data,
        actions: List[Action],
        logp: Tensor,
        pis: List[ActionConditionalDistribution],
    ):
        self.final_data = final_data
        self.actions = actions
        self.logp = logp
        self.pis = pis

    def __len__(self):
        return len(self.actions)


class FailedEpisode(Episode):
    """This will be handled separately by the reward function"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FailedCompleteEpisode(FailedEpisode):  # self.actions[-1] is a StopAction
    def __str__(self):
        return f"FailedCompleteEpisode(n={len(self)})"


class FailedIntermediateEpisode(FailedEpisode):
    def __init__(self, *args, failure_msg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.failure_msg = failure_msg

    def __str__(self):
        return f"FailedIntermediateEpisode(n={len(self)})"


class SuccessfulEpisode(Episode):
    def __init__(self, obj, *args, **kwargs):
        """
        Args:
            obj (float): rdkit mol object or SimplePlanarTree object, which will be used to compute the reward
        """
        super().__init__(*args, **kwargs)
        self.obj = obj

    def __str__(self):
        return f"SuccessfulEpisode(n={len(self)})"

    def get_obj_length(self):
        if isinstance(self.obj, RdkitMol):
            return self.obj.GetNumAtoms()
        elif hasattr(self.obj, "__len__"):
            return len(self.obj)
        else:
            raise NotImplementedError(type(self.obj))


class PreEpisode(metaclass=ABCMeta):
    def __init__(
        self, featurize_module, atom_composer, data, actions=[], logps=[], pis=[]
    ):
        self.data = data
        self.actions: List[Action] = actions
        self.logps: List[Tensor] = logps
        self.pis: List[ActionConditionalDistribution] = pis

        self.featurize_module = featurize_module
        self.atom_composer = atom_composer

    def update(self, action, logp, pi):
        self.actions.append(action)
        self.logps.append(logp)
        self.pis.append(pi)

        if not isinstance(action, StopAction):
            self.data = self.atom_composer(
                get_data_with_action_applied(self.data, action, self.featurize_module)
            )

    def clone(self) -> "PreEpisode":
        data = self.data.clone()
        actions = deepcopy(self.actions)
        logps = deepcopy(self.logps)
        pis = deepcopy(self.pis)
        return PreEpisode(
            self.featurize_module,
            self.atom_composer,
            data,
            actions=actions,
            logps=logps,
            pis=pis,
        )

    def to_episode(
        self, obj, intermediate=False, failure_msg=None, **kwargs
    ) -> Episode:
        assert self.logps[0].ndim == 0
        logp = torch.stack(self.logps)
        if obj is None:
            if intermediate:
                return FailedIntermediateEpisode(
                    self.data,
                    self.actions,
                    self.logps,
                    self.pis,
                    failure_msg=failure_msg,
                    **kwargs,
                )
            else:
                return FailedCompleteEpisode(
                    self.data, self.actions, logp, self.pis, **kwargs
                )
        else:
            assert not intermediate
            return SuccessfulEpisode(
                obj, self.data, self.actions, logp, self.pis, **kwargs
            )
