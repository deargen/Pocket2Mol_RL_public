from pocket2mol_rl.evaluation.metric.base import Metric
from typing import List, Dict, Tuple, Union, Any, Optional
from rdkit.Chem.rdchem import Mol as RdkitMol
from rdkit.Chem import MolToSmiles
from pathlib import Path
from pocket2mol_rl import ROOT_DIR
from .modules.energy_ratio import check_energy_ratio, CheckEnergyRatioException
from .modules.flatness import check_flatness, CheckFlatnessException
from .modules.distance_geometry import check_geometry, CheckGeometryException
import torch
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from pocket2mol_rl.utils.tensor import nonnan_running_average
from math import ceil

from pocket2mol_rl.utils.silence import silence_rdkit

silence_rdkit()

DEFAULT_ENSEMBLE_NUMBER_CONFORMATIONS = 4
DEFAULT_THRESHOLD_ENERGY_RATIO = 50


class EnergyRatio(Metric):
    def __init__(
        self,
        ensemble_number_conformations=DEFAULT_ENSEMBLE_NUMBER_CONFORMATIONS,
        threshold_energy_ratio=DEFAULT_THRESHOLD_ENERGY_RATIO,
        upper_bound=100.0,
    ):
        self.ensemble_number_conformations = ensemble_number_conformations
        self.threshold_energy_ratio = threshold_energy_ratio
        self.upper_bound = upper_bound

    @property
    def bigger_is_better(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "energy-ratio"

    def __repr__(self):
        return self.name

    @classmethod
    def parse_metric(cls, s: str) -> Optional["EnergyRatio"]:
        if s == "energy-ratio":
            return cls()
        return None

    def _get_energy_ratio_results(self, mol: RdkitMol):
        try:
            results = check_energy_ratio(
                mol,
                ensemble_number_conformations=self.ensemble_number_conformations,
                threshold_energy_ratio=self.threshold_energy_ratio,
                raise_error=True,
            )
        except Exception as e:
            raise CheckEnergyRatioException() from e
        except BaseException as e:
            raise e

        return results["results"]

    def evaluate(self, mol: RdkitMol, protein_file: Path) -> Optional[float]:
        try:
            results = self._get_energy_ratio_results(mol)
            energy_ratio = results["energy_ratio"]
            if np.isfinite(energy_ratio):
                energy_ratio = min(energy_ratio, self.upper_bound)
            return energy_ratio
        except CheckEnergyRatioException:
            return None
        except BaseException as e:
            raise e


class EnergyRatioPass(EnergyRatio):
    def __init__(
        self,
        ensemble_number_conformations=DEFAULT_ENSEMBLE_NUMBER_CONFORMATIONS,
        threshold_energy_ratio=DEFAULT_THRESHOLD_ENERGY_RATIO,
    ):
        super().__init__(
            ensemble_number_conformations=ensemble_number_conformations,
            threshold_energy_ratio=threshold_energy_ratio,
        )

    @property
    def bigger_is_better(self) -> bool:
        return True

    @property
    def name(self) -> str:
        if self.threshold_energy_ratio == DEFAULT_THRESHOLD_ENERGY_RATIO:
            return f"energy-ratio-pass"
        else:
            return f"energy-ratio-pass-{self.threshold_energy_ratio}"

    @classmethod
    def parse_metric(cls, s: str) -> Optional["EnergyRatioPass"]:
        if s.startswith("energy-ratio-pass"):
            if s == "energy-ratio-pass":
                threshold_energy_ratio = DEFAULT_THRESHOLD_ENERGY_RATIO
            else:
                threshold_energy_ratio = int(s.split("-")[-1])
            return cls(threshold_energy_ratio=threshold_energy_ratio)
        return None

    def evaluate(self, mol: RdkitMol, protein_file: Path) -> Optional[float]:
        try:
            results = self._get_energy_ratio_results(mol)
            passed = results["energy_ratio_passes"]
            assert isinstance(passed, bool)
            return float(passed)
        except CheckEnergyRatioException:
            return None
        except BaseException as e:
            raise e


class BustPass(EnergyRatio):
    def __init__(
        self,
        ensemble_number_conformations=4,
        threshold_energy_ratio=DEFAULT_THRESHOLD_ENERGY_RATIO,
    ):
        super().__init__(
            ensemble_number_conformations=ensemble_number_conformations,
            threshold_energy_ratio=threshold_energy_ratio,
        )

    @property
    def bigger_is_better(self) -> bool:
        return True

    @property
    def name(self) -> str:
        if self.threshold_energy_ratio == DEFAULT_THRESHOLD_ENERGY_RATIO:
            return f"bust-pass"
        else:
            return f"bust-pass-{self.threshold_energy_ratio}"

    @classmethod
    def parse_metric(cls, s: str) -> Optional["EnergyRatioPass"]:
        if s.startswith("bust-pass"):
            if s == "bust-pass":
                threshold_energy_ratio = DEFAULT_THRESHOLD_ENERGY_RATIO
            else:
                threshold_energy_ratio = int(s.split("-")[-1])
            return cls(threshold_energy_ratio=threshold_energy_ratio)
        return None

    def _get_flatness_results(self, mol: RdkitMol):
        try:
            flatness_results = check_flatness(mol, raise_error=True)
        except Exception as e:
            raise CheckFlatnessException() from e
        except BaseException as e:
            raise e

        return flatness_results["results"]

    def _get_geometry_results(self, mol: RdkitMol):
        try:
            distance_geometry_results = check_geometry(mol, raise_error=True)
        except Exception as e:
            raise CheckGeometryException() from e
        except BaseException as e:
            raise e

        return distance_geometry_results["results"]

    def evaluate(self, mol: RdkitMol, protein_file: Path) -> Optional[float]:
        try:
            energy_ratio_results = self._get_energy_ratio_results(mol)
            energy_ratio_passed = energy_ratio_results["energy_ratio_passes"]
        except CheckEnergyRatioException:
            return None
        except BaseException as e:
            raise e
        assert isinstance(energy_ratio_passed, bool)
        if not energy_ratio_passed:
            return 0.0

        try:
            flatness_results = self._get_flatness_results(mol)
            flatness_passed = flatness_results["flatness_passes"]
        except CheckFlatnessException:
            return None
        except BaseException as e:
            raise e
        assert isinstance(flatness_passed, bool)
        if not flatness_passed:
            return 0.0

        """
        try:
            geometry_results = self._get_geometry_results(mol)
            geometry_passed = geometry_results["geometry_passes"]
        except CheckGeometryException:
            return None
        except BaseException as e:
            raise e
        assert isinstance(geometry_passed, bool)
        if not geometry_passed:
            return 0.0
        """

        return 1.0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_ratio(a, b, default=1.0):
    assert a >= 0
    assert b >= 0
    if b == 0:
        return default
    return a / b


class BustScore(EnergyRatio):
    def __init__(
        self,
        ensemble_number_conformations=4,
        flatnass_score_default=0.5,
        clash_score_default=0.5,
        rescaled=False,
    ):
        super().__init__(
            ensemble_number_conformations=ensemble_number_conformations,
        )

        self.flatness_score_default = flatnass_score_default
        self.clash_score_default = clash_score_default

        self.rescaled = rescaled

    @property
    def bigger_is_better(self) -> bool:
        return True

    @property
    def name(self) -> str:
        if self.rescaled:
            return f"bust-score-rescaled"
        else:
            return f"bust-score"

    @classmethod
    def parse_metric(cls, s: str) -> Optional["EnergyRatioPass"]:
        if s == "bust-score":
            return cls()
        elif s == "bust-score-rescaled":
            return cls(rescaled=True)
        return None

    def _get_flatness_results(self, mol: RdkitMol):
        try:
            flatness_results = check_flatness(mol, raise_error=True)
        except Exception as e:
            raise CheckFlatnessException() from e
        except BaseException as e:
            raise e

        return flatness_results["results"]

    def _get_geometry_results(self, mol: RdkitMol):
        try:
            distance_geometry_results = check_geometry(mol, raise_error=True)
        except Exception as e:
            raise CheckGeometryException() from e
        except BaseException as e:
            raise e

        return distance_geometry_results["results"]

    def _get_flatness_score(self, mol: RdkitMol):
        try:
            flatness_results = self._get_flatness_results(mol)
            num_systems_checked = flatness_results["num_systems_checked"]
            num_systems_passed = flatness_results["num_systems_passed"]
            flatness_score = get_ratio(
                num_systems_passed,
                num_systems_checked,
                default=self.flatness_score_default,
            )
        except CheckFlatnessException:
            return None
        except BaseException as e:
            raise e
        return flatness_score

    def _get_geometry_scores(self, mol: RdkitMol):
        try:
            geometry_results = self._get_geometry_results(mol)
            num_bonds = geometry_results["number_bonds"]
            num_valid_bonds = geometry_results["number_valid_bonds"]
            num_angles = geometry_results["number_angles"]
            num_valid_angles = geometry_results["number_valid_angles"]
            num_noncov_pairs = geometry_results["number_noncov_pairs"]
            num_valid_noncov_pairs = geometry_results["number_valid_noncov_pairs"]
            num_clashes = geometry_results["number_clashes"]
            assert num_clashes == num_noncov_pairs - num_valid_noncov_pairs

            bond_score = get_ratio(num_valid_bonds, num_bonds, default=0)
            angle_score = get_ratio(num_valid_angles, num_angles, default=0)
            if num_noncov_pairs == 0:
                clash_score = self.clash_score_default
            else:
                clash_score = float(num_clashes == 0)

        except CheckGeometryException:
            return None
        except BaseException as e:
            raise e

        return {
            "bond_score": bond_score,
            "angle_score": angle_score,
            "clash_score": clash_score,
        }

    def _get_energy_ratio_score(self, mol: RdkitMol):
        try:
            energy_ratio_results = self._get_energy_ratio_results(mol)
            energy_ratio = energy_ratio_results["energy_ratio"]
            log10_energy_ratio = np.log10(abs(energy_ratio) + 1e-6)
            # 1 -> 3
            # 2 -> -2
            rescaled_log10_energy_ratio = 8 - 5 * log10_energy_ratio
            energy_ratio_score = sigmoid(rescaled_log10_energy_ratio)
            """ 
            energy_ratio -> energy_ratio_score:
            1 -> 0.9997
            10 -> 0.95
            50 -> 0.38
            100 -> 0.12
            200 -> 0.03
            """
        except CheckEnergyRatioException:
            return None
        except BaseException as e:
            raise e
        return energy_ratio_score

    def evaluate(self, mol: RdkitMol, protein_file: Path) -> Optional[float]:
        flatness_score = self._get_flatness_score(mol)
        if flatness_score is None or flatness_score == 0:
            return flatness_score

        geometry_scores = self._get_geometry_scores(mol)
        if geometry_scores is None:
            return None
        bond_score = geometry_scores["bond_score"]
        angle_score = geometry_scores["angle_score"]
        clash_score = geometry_scores["clash_score"]
        if min(bond_score, angle_score, clash_score) == 0:
            return 0.0

        energy_ratio_score = self._get_energy_ratio_score(mol)
        if energy_ratio_score is None:
            return None

        if self.rescaled:
            # TODO: change the constants later. intention: the final "min" can be determined by this often enough
            # TODO: make the constants as kwargs, and make them configurable from config files
            flatness_score = max(0, 1 + (flatness_score - 1) * 1.85)
            bond_score = max(0, 1 + (bond_score - 1) * 5.34)
            angle_score = max(0, 1 + (angle_score - 1) * 25.95)

            clash_score = max(0, 1 + (clash_score - 1) * 5.79)

        return min(
            flatness_score, bond_score, angle_score, clash_score, energy_ratio_score
        )


class RingFlatnessScore(BustScore):
    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return f"flatness-score"

    @classmethod
    def parse_metric(cls, s: str) -> Optional["EnergyRatioPass"]:
        if s == "flatness-score":
            return cls()
        return None

    def evaluate(self, mol: RdkitMol, protein_file: Path) -> Optional[float]:
        return self._get_flatness_score(mol)


class BondScore(BustScore):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return f"bond-score"

    @classmethod
    def parse_metric(cls, s: str) -> Optional["EnergyRatioPass"]:
        if s == "bond-score":
            return cls()
        return None

    def evaluate(self, mol: RdkitMol, protein_file: Path) -> Optional[float]:
        geometry_scores = self._get_geometry_scores(mol)
        if geometry_scores is None:
            return None
        assert "bond_score" in geometry_scores, geometry_scores
        return geometry_scores["bond_score"]


class AngleScore(BustScore):
    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return f"angle-score"

    @classmethod
    def parse_metric(cls, s: str) -> Optional["EnergyRatioPass"]:
        if s == "angle-score":
            return cls()
        return None

    def evaluate(self, mol: RdkitMol, protein_file: Path) -> Optional[float]:
        geometry_scores = self._get_geometry_scores(mol)
        if geometry_scores is None:
            return None
        assert "angle_score" in geometry_scores, geometry_scores
        return geometry_scores["angle_score"]


class ClashScore(BustScore):
    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return f"clash-score"

    @classmethod
    def parse_metric(cls, s: str) -> Optional["EnergyRatioPass"]:
        if s == "clash-score":
            return cls()
        return None

    def evaluate(self, mol: RdkitMol, protein_file: Path) -> Optional[float]:
        geometry_scores = self._get_geometry_scores(mol)
        if geometry_scores is None:
            return None
        assert "clash_score" in geometry_scores, geometry_scores
        return geometry_scores["clash_score"]


class EnergyRatioScore(BustScore):
    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return f"energy-ratio-score"

    @classmethod
    def parse_metric(cls, s: str) -> Optional["EnergyRatioPass"]:
        if s == "energy-ratio-score":
            return cls()
        return None

    def evaluate(self, mol: RdkitMol, protein_file: Path) -> Optional[float]:
        return self._get_energy_ratio_score(mol)
