from pathlib import Path
from typing import Union

import numpy as np
from Bio.PDB.PDBParser import PDBParser


def get_center_of_pdb_file(pdb_file: Union[str, Path]):
    if type(pdb_file) != Path:
        pdb_file = Path(pdb_file)
    if not pdb_file.exists():
        raise FileNotFoundError(pdb_file)

    structure = PDBParser(PERMISSIVE=True, QUIET=True).get_structure("", pdb_file)
    coords = np.stack([atom.get_coord() for atom in structure.get_atoms()])
    return coords.mean(axis=0).astype(dtype=np.float64)


def get_radius_of_pdb_file(pdb_file: Union[str, Path]):
    if type(pdb_file) != Path:
        pdb_file = Path(pdb_file)
    if not pdb_file.exists():
        raise FileNotFoundError(pdb_file)

    structure = PDBParser(PERMISSIVE=True, QUIET=True).get_structure("", pdb_file)
    coords = np.stack([atom.get_coord() for atom in structure.get_atoms()])
    center = coords.mean(axis=0)
    radius = np.linalg.norm(coords - center, axis=1).max()

    return radius


def get_bbox_size_for_pdb_file(pdb_file: Union[str, Path]):
    if type(pdb_file) != Path:
        pdb_file = Path(pdb_file)
    if not pdb_file.exists():
        raise FileNotFoundError(pdb_file)

    structure = PDBParser(PERMISSIVE=True, QUIET=True).get_structure("", pdb_file)
    coords = np.stack([atom.get_coord() for atom in structure.get_atoms()])
    center = coords.mean(axis=0)
    bbox_size = np.max(np.abs(coords - center)) * 2

    return bbox_size
