from rdkit import Chem
from rdkit.Chem.rdchem import Mol as RdkitMol
from pathlib import Path
import numpy as np


def parse_sdf(sdf_file: Path) -> RdkitMol:
    """
    Parse sdf file into a list of rdkit mol objects
    Args:
        sdf_file: path to the sdf file
    Returns:
        list of rdkit mol objects
    """
    sdf_file = Path(sdf_file)
    if not sdf_file.exists():
        raise FileNotFoundError(f"{sdf_file} does not exist.")
    suppl = Chem.SDMolSupplier(str(sdf_file))
    mols = [mol for mol in suppl if mol is not None]
    assert len(mols) == 1
    return mols[0]


def write_sdf(mol, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def get_center_of_sdf_file(sdf_file: Path) -> list:
    """
    Get the center of a sdf file
    Args:
        sdf_file: path to the sdf file
    Returns:
        center: center of the sdf file
    """
    mol = parse_sdf(sdf_file)
    coords = mol.GetConformer().GetPositions()
    coords_mean = np.mean(coords, axis=0)
    return coords_mean.tolist()
