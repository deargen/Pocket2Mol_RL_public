from pocket2mol_rl.evaluation.metric.base import Metric

from typing import List, Dict, Tuple, Union, Any, Optional
from rdkit.Chem.rdchem import Mol as RdkitMol
from rdkit.Chem.QED import qed
from pathlib import Path


class MoleculeLength(Metric):
    @property
    def name(self) -> str:
        return "length"

    def __repr__(self):
        return "length"

    @classmethod
    def parse_metric(cls, s: str) -> Optional["MoleculeLength"]:
        if s == "length":
            return MoleculeLength()
        return None

    def evaluate(self, mol: RdkitMol, protein_file: Path) -> float:
        return mol.GetNumAtoms()

    @property
    def bigger_is_better(self) -> bool:
        return None

    @property
    def discrete(self):
        return True


class QEDScore(Metric):
    @property
    def name(self) -> str:
        return "qed"

    def __repr__(self):
        return "qed"

    @classmethod
    def parse_metric(cls, s: str) -> Optional["QEDScore"]:
        if s == "qed":
            return QEDScore()
        return None

    def evaluate(self, mol: RdkitMol, protein_file: Path) -> float:
        try:
            return qed(mol)
        except:
            return None

    @property
    def bigger_is_better(self) -> bool:
        return True

    @property
    def discrete(self):
        return False
