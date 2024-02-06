from abc import ABCMeta, abstractclassmethod, abstractmethod, abstractproperty
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from rdkit.Chem.rdchem import Mol as RdkitMol
from tqdm import tqdm


class Metric(metaclass=ABCMeta):
    @abstractproperty
    def name(self) -> str:
        """
        Note:
            mlflow does not allow metric names to contain some special characters.
            Names may only contain alphanumerics, underscores (_), dashes (-),
            periods (.), spaces ( ), and slashes (/).
        """
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractclassmethod
    def parse_metric(cls, s: str) -> Optional["Metric"]:
        """Try to initialize a metric object from a string. Return None if not applicable."""
        pass

    @abstractmethod
    def evaluate(self, mol: RdkitMol, protein_file: Path) -> Optional[float]:
        pass

    # Overridden in some subclasses for efficiency
    def evaluate_in_batch(
        self,
        mols: List[RdkitMol],
        protein_file: Path,
        pbar=False,
    ) -> List[Optional[float]]:
        if pbar:
            mols = tqdm(mols)
        return [self.evaluate(mol, protein_file) for mol in mols]

    @abstractproperty
    def bigger_is_better(self) -> bool:
        pass

    @property
    def compare_with_baseline(self) -> bool:
        return True

    @property
    def digit(self):
        return 3

    @property
    def unit(self):
        return ""

    @property
    def discrete(self):
        return False
