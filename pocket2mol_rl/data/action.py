import math
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from pocket2mol_rl.utils.deterministic import assign_val_at_idx

GAUSSIAN_COEF = 1.0 / math.sqrt(2 * math.pi)


class Action:
    def allclose(self, other: "Action", atol=1e-8, rtol=1e-5) -> bool:
        raise NotImplementedError()


class StopAction(Action):
    def to(self, device):
        return StopAction()

    @property
    def device(self):
        return "cpu"


class FocalStopAction(StopAction):
    def __str__(self):
        return "FocalStopAction()"

    def to(self, device):
        return FocalStopAction()

    def allclose(self, other: Action, atol=1e-8, rtol=1e-5) -> bool:
        return isinstance(other, FocalStopAction)


@dataclass
class AtomAction(Action):
    idx_focal: Optional[Tensor]
    pos: Tensor
    element: Tensor
    bond_index: Optional[Tensor]
    bond_type: Optional[Tensor]

    def __str__(self):
        return f"AtomAction(idx_focal={self.idx_focal}, pos={self.pos}, element={self.element}, bond_index={self.bond_index}, bond_type={self.bond_type})"

    @property
    def device(self):
        return self.pos.device

    def to(self, device):
        return AtomAction(
            idx_focal=self.idx_focal.to(device) if self.idx_focal is not None else None,
            pos=self.pos.to(device),
            element=self.element.to(device),
            bond_index=(
                self.bond_index.to(device) if self.bond_index is not None else None
            ),
            bond_type=self.bond_type.to(device) if self.bond_type is not None else None,
        )

    def allclose(self, other: Action, atol=1e-8, rtol=1e-5) -> bool:
        if not isinstance(other, AtomAction):
            return False
        attrs = ["idx_focal", "pos", "element", "bond_index", "bond_type"]
        for attr in attrs:
            a = getattr(self, attr)
            b = getattr(other, attr)
            if a is None:
                if b is not None:
                    return False
            else:
                if b is None:
                    return False
                if a.shape != b.shape:
                    return False
                try:
                    if not torch.allclose(a, b, atol=atol, rtol=rtol):
                        return False
                except:
                    pass

        return True


class Distribution(metaclass=ABCMeta):
    @abstractmethod
    def sample(self, seed=None, generator=None) -> Tensor:
        pass

    @abstractmethod
    def get_logp(self, val: Tensor) -> Tensor:
        pass

    @abstractmethod
    def allclose(self, other: "Distribution", atol=1e-8, rtol=1e-5) -> bool:
        pass


class IdxFocalDistribution(Distribution):
    def __init__(
        self,
        y_frontier_pred: Tensor,
        allow_stop=True,
        stop_first=False,
        focalizable_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            y_frontier_pred (Tensor): 1D tensor of shape (context_atoms,) containing logits for each context atom ("context" can mean either ligand context or protein context)
        TODO: option 1: nonzero logit for None (as parameter) option 2: add threshold
        """
        assert y_frontier_pred.ndim == 1
        if focalizable_mask is not None:
            assert focalizable_mask.ndim == 1
            assert len(focalizable_mask) == len(y_frontier_pred)
            assert focalizable_mask.dtype == torch.bool
            assert torch.any(focalizable_mask)
        self.allow_stop = allow_stop
        self.stop_first = stop_first
        if self.allow_stop:
            if self.stop_first:
                continue_prob = torch.sigmoid(torch.max(y_frontier_pred))
                self.probs = torch.empty(
                    len(y_frontier_pred) + 1,
                    device=y_frontier_pred.device,
                    dtype=y_frontier_pred.dtype,
                )
                self.probs[:-1] = torch.softmax(y_frontier_pred, dim=0) * continue_prob
                self.probs[-1] = 1 - continue_prob
            else:
                self.probs = torch.softmax(
                    torch.cat(
                        [y_frontier_pred, torch.zeros(1, device=y_frontier_pred.device)]
                    ),
                    dim=0,
                )

            if focalizable_mask is not None:
                focalizable_mask = torch.cat(
                    [
                        focalizable_mask,
                        torch.tensor([True], device=focalizable_mask.device),
                    ]
                )

        else:
            self.probs = torch.softmax(y_frontier_pred, dim=0)

        if focalizable_mask is not None:
            # condition on focalizable_mask
            probs = torch.zeros_like(self.probs)
            unnormalized_conditioned = self.probs[focalizable_mask] + 1e-10
            probs[focalizable_mask] = (
                unnormalized_conditioned / unnormalized_conditioned.sum()
            )
            self.probs = probs

        assert torch.abs(self.probs.sum() - 1) < 1e-3

    def __str__(self):
        return f"IdxFocalDistribution(probs={self.probs})"

    def sample(self, hard_stop=None, seed=None, generator=None) -> Optional[Tensor]:
        if generator is None:
            if seed is not None:
                generator = torch.Generator(device=self.probs.device).manual_seed(seed)

        if self.allow_stop:
            if hard_stop is None:
                i = self.probs.multinomial(1, generator=generator)[0]
                if i == len(self.probs) - 1:
                    return None
                else:
                    return i
            else:
                assert 0 < hard_stop < 1
                if self.probs[-1] > hard_stop:
                    return None
                probs = self.probs[:-1] / (1 - self.probs[-1])

                i = probs.multinomial(1, generator=generator)[0]
                return i
        else:
            i = self.probs.multinomial(1, generator=generator)[0]
            return i

    def get_logp(self, val: Optional[Tensor], eps=1e-6) -> Tensor:
        if val is None:
            assert self.allow_stop
            return torch.log(self.probs[-1] + eps)
        else:
            assert val.dtype == torch.long
            return torch.log(self.probs[val] + eps)

    def allclose(self, other: "IdxFocalDistribution", atol=1e-8, rtol=1e-5) -> bool:
        return (
            self.allow_stop == other.allow_stop
            and self.stop_first == other.stop_first
            and self.probs.shape == other.probs.shape
            and torch.allclose(self.probs, other.probs, atol=atol, rtol=rtol)
        )


class PosDistribution(Distribution):
    def __init__(self, mu: Tensor, sigma: Tensor, pi: Tensor):
        """
        Args:
            mu (Tensor): modes of the gaussian mixture distribution, of shape (n_component, space_dim)
            sigma (Tensor): standard deviations of the gaussian mixture distribution, of shape (n_component, space_dim)
            pi (Tensor): mixture weights of the gaussian mixture distribution, of shape (n_component,). Sums to 1.
        """
        self.n_component, self.space_dim = mu.shape
        assert sigma.shape == (self.n_component, self.space_dim)
        assert pi.shape == (self.n_component,)

        self.mu = mu
        self.sigma = sigma + 1e-16
        self.pi = pi

    def __str__(self):
        return f"PosDistribution(mu={self.mu}, sigma={self.sigma}, pi={self.pi})"

    def sample(self, only_mean=False, seed=None, generator=None) -> Tensor:
        if generator is None:
            if seed is not None:
                generator = torch.Generator(device=self.probs.device).manual_seed(seed)
        idx = self.pi.multinomial(1, generator=generator)[0]
        if only_mean:
            return self.mu[idx]
        else:
            z = torch.randn(
                self.mu.shape[1:], device=self.mu.device, generator=generator
            )
            return self.mu[idx] + self.sigma[idx] * z

    def get_pdf(self, pos: Tensor) -> Tensor:
        if pos.ndim == 1:
            assert pos.shape == (self.space_dim,)
            target = pos.unsqueeze(0)
            errors = target - self.mu
            p = (
                GAUSSIAN_COEF
                * torch.exp(-0.5 * (errors / self.sigma) ** 2)
                / self.sigma
            )
            p = torch.prod(p, dim=1)
            return torch.sum(self.pi * p)
        elif pos.ndim == 2:
            assert pos.shape[1] == self.space_dim
            target = pos.unsqueeze(1)
            mu = self.mu.unsqueeze(0)
            sigma = self.sigma.unsqueeze(0)
            pi = self.pi.unsqueeze(0)
            errors = target - mu
            p = GAUSSIAN_COEF * torch.exp(-0.5 * (errors / sigma) ** 2) / sigma
            p = torch.prod(p, dim=2)
            return torch.sum(pi * p, dim=1)
        else:
            raise ValueError(f"pos.ndim must be 1 or 2, but got {pos.ndim}")

    def get_logp(self, pos: Tensor, eps=1e-6) -> Tensor:
        return torch.log(self.get_pdf(pos) + eps)

    def allclose(self, other: "PosDistribution", atol=1e-8, rtol=1e-5) -> bool:
        return (
            self.mu.shape == other.mu.shape
            and torch.allclose(self.mu, other.mu, atol=atol, rtol=rtol)
            and torch.allclose(self.sigma, other.sigma, atol=atol, rtol=rtol)
            and torch.allclose(self.pi, other.pi, atol=atol, rtol=rtol)
        )


class ElementDistribution(Distribution):
    def __init__(self, y_query_pred: Tensor):
        """
        Args:
            y_query_pred (Tensor): 1D tensor of shape (element_type,) containing logits for each element type
        """
        self.probs = torch.softmax(y_query_pred, dim=0)

    def __str__(self):
        return f"ElementDistribution(probs={self.probs})"

    def sample(self, seed=None, generator=None) -> Tensor:
        if generator is None:
            if seed is not None:
                generator = torch.Generator(device=self.probs.device).manual_seed(seed)
        return self.probs.multinomial(1, generator=generator)[0]

    def get_logp(self, element: Tensor, eps=1e-6) -> Tensor:
        assert element.dtype == torch.long
        return torch.log(self.probs[element] + eps)

    def allclose(self, other: "ElementDistribution", atol=1e-8, rtol=1e-5) -> bool:
        return self.probs.shape == other.probs.shape and torch.allclose(
            self.probs, other.probs, atol=atol, rtol=rtol
        )


class BondDistribution(Distribution):
    def __init__(self, edge_pred: Tensor):
        """
        Args:
            edge_pred (Tensor): 2D tensor of shape (ligand_context_atoms, bond_type) containing logits for each (ligand_contex atom, bond type) pair
        """
        self.n = len(edge_pred)
        self.probs = torch.softmax(edge_pred, dim=1)
        pass

    def __str__(self):
        return f"BondDistribution(probs={self.probs})"

    def sample(self, seed=None, generator=None) -> Tuple[Tensor, Tensor]:
        if generator is None:
            if seed is not None:
                generator = torch.Generator(device=self.probs.device).manual_seed(seed)
        all_edge_type = self.probs.multinomial(1, generator=generator).squeeze(dim=1)
        bond_mask = all_edge_type > 0

        bond_index = torch.arange(len(all_edge_type), device=bond_mask.device)[
            bond_mask
        ]
        bond_index = torch.stack(
            [self.n * torch.ones_like(bond_index), bond_index], dim=0
        )
        bond_type = all_edge_type[bond_mask]
        return (bond_index, bond_type)

    def get_logp(self, bond: Tuple[Tensor, Tensor], eps=1e-6) -> Tensor:
        bond_index, bond_type = bond
        assert torch.all(bond_index[0] == self.n)
        idxs = bond_index[1]
        bond_mask = torch.zeros(self.n, dtype=torch.bool, device=idxs.device)

        assign_val_at_idx(bond_mask, idxs, True)
        # `bond_mask[idxs] = True`` errors when torch.use_deterministic_algorithms(True)

        bond_probs = self.probs[idxs, bond_type]
        nonbond_probs = self.probs[~bond_mask, 0]
        return torch.log(bond_probs + eps).sum() + torch.log(nonbond_probs + eps).sum()

    def allclose(self, other: "BondDistribution", atol=1e-8, rtol=1e-5) -> bool:
        return self.probs.shape == other.probs.shape and torch.allclose(
            self.probs, other.probs, atol=atol, rtol=rtol
        )


class ActionConditionalDistribution:
    def __init__(
        self,
        condition: Action,
        pi_idx_focal: IdxFocalDistribution,
        pi_pos: Optional[PosDistribution],  # may be None for FocalStopAction
        pi_element: Optional[ElementDistribution],  # may be None for FocalStopAction
        pi_bond: Optional[BondDistribution],
    ):
        """
        Args:
            condition (Action)
            pi_idx_focal (IdxFocalDistribution):
            pi_pos (PosDistribution):
            pi_element (ElementDistribution:
            pi_bond (BondDistribution):
        """
        self.condition = condition
        self.pi_idx_focal = pi_idx_focal
        self.pi_pos = pi_pos
        self.pi_element = pi_element
        self.pi_bond = pi_bond

    def get_logp(self, action: Action) -> Tensor:
        if isinstance(action, StopAction):
            assert isinstance(action, FocalStopAction)
            logp_idx_focal = self.pi_idx_focal.get_logp(None)
            return logp_idx_focal
        logp_idx_focal = self.pi_idx_focal.get_logp(action.idx_focal)
        logp = logp_idx_focal
        logp_pos = self.pi_pos.get_logp(action.pos)
        logp = logp + logp_pos
        logp_element = self.pi_element.get_logp(action.element)
        logp = logp + logp_element
        if self.pi_bond is not None:
            logp_bond = self.pi_bond.get_logp((action.bond_index, action.bond_type))
            logp = logp + logp_bond
        return logp

    def allclose(
        self, other: "ActionConditionalDistribution", atol=1e-8, rtol=1e-5
    ) -> bool:
        if not self.condition.allclose(other.condition, atol=atol, rtol=rtol):
            return False

        attrs = ["pi_idx_focal", "pi_pos", "pi_element", "pi_bond"]
        for attr in attrs:
            a = getattr(self, attr)
            b = getattr(other, attr)
            if a is None:
                if b is not None:
                    return False
            else:
                if b is None:
                    return False
                if not a.allclose(b, atol=atol, rtol=rtol):
                    return False

        return True
