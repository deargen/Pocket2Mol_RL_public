import copy

import numpy as np
import torch

# from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader


import warnings

from Bio import BiopythonWarning
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Selection import unfold_entities
from easydict import EasyDict
from rdkit import Chem

from pocket2mol_rl.data.protein_ligand import PDBProtein
from pocket2mol_rl.data.data import ProteinLigandData
from pathlib import Path

from abc import ABCMeta, abstractmethod
from pocket2mol_rl.utils.mol import parse_sdf
from typing import Optional, List

from pocket2mol_rl.data.transform.featurize import (
    FeaturizeProteinAtom,
    FeaturizeLigandAtom,
)
from pocket2mol_rl.data.transform import (
    AtomComposer,
    LigandMaskAll,
    LigandMaskNone,
    LigandCountNeighbors,
)
from torch_geometric.transforms import Compose


from easydict import EasyDict

DEFAULT_MODEL_CONFIG_FOR_FEATURIZER = EasyDict(
    {
        "encoder": {
            "knn": 48,
            "num_edge_types": 4,
        }
    }
)


def _process_data_from_scratch(data, model_config=None, mask="all"):
    if model_config is None:
        model_config = DEFAULT_MODEL_CONFIG_FOR_FEATURIZER
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    if mask == "all":
        mask = LigandMaskAll(featurize_module=ligand_featurizer)
    elif mask == "none":
        mask = LigandMaskNone(featurize_module=ligand_featurizer)
    else:
        raise ValueError(mask)
    atom_composer = AtomComposer(
        model_config.encoder.knn, model_config.encoder.num_edge_types
    )
    transform = Compose(
        [
            LigandCountNeighbors(),
            protein_featurizer,
            ligand_featurizer,
            mask,
            atom_composer,
        ]
    )
    data = transform(data)
    return data


class CoordinateFilter(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, pos: torch.Tensor) -> bool:
        pass

    @abstractmethod
    def get_message_when_no_atom_found(self):
        pass

    def filter_and_save_pdb(self, src_pdb: Path, tgt_pdb: Path):
        if not src_pdb.exists():
            raise FileNotFoundError(src_pdb)
        if tgt_pdb.exists():
            print(f"Skip saving {tgt_pdb} because it already exists.")
            return
        tgt_pdb.parent.mkdir(exist_ok=True, parents=True)

        warnings.simplefilter("ignore", BiopythonWarning)
        ptable = Chem.GetPeriodicTable()
        parser = PDBParser()
        model = parser.get_structure(None, str(src_pdb))[0]
        model_copy = copy.deepcopy(model)
        # filter all atoms
        for atom in unfold_entities(model_copy, "A"):
            res = atom.get_parent()
            resname = res.get_resname()
            if resname == "MSE":
                resname = "MET"
            if resname not in PDBProtein.AA_NAME_NUMBER:
                continue  # Ignore water, heteros, and non-standard residues.

            element_symb = atom.element.capitalize()
            if element_symb == "H":
                continue
            x, y, z = atom.get_coord()
            pos = torch.FloatTensor([x, y, z])

            if not self.__call__(pos):
                atom.parent.detach_child(atom.get_id())
        # save the filtered pdb
        io = PDBIO()
        io.set_structure(model_copy)
        io.save(str(tgt_pdb))


class NoFilter(CoordinateFilter):
    def __call__(self, pos: torch.Tensor) -> bool:
        return True

    def get_message_when_no_atom_found(self):
        return f"No atoms found."


class BoxCoordinateFilter(CoordinateFilter):
    def __init__(self, center, bbox_size):
        assert len(center) == 3
        self.center = torch.FloatTensor(center)
        assert bbox_size > 0
        self.bbox_size = bbox_size

    def __call__(self, pos: torch.Tensor) -> bool:
        return (pos - self.center).abs().max() <= (self.bbox_size / 2)

    def get_message_when_no_atom_found(self):
        return f"No atoms found in the bounding box (center={self.center}, size={self.bbox_size})."


class RefSpanCoordinateFilter(CoordinateFilter):
    def __init__(self, ref_sdf_path: Path, distance_threshold: float):
        mol = parse_sdf(ref_sdf_path)
        self.ref_coords = torch.from_numpy(mol.GetConformer().GetPositions()).to(
            dtype=torch.float32
        )
        n = len(self.ref_coords)
        assert self.ref_coords.shape == (n, 3)

        assert distance_threshold >= 1.0
        self.distance_threshold = distance_threshold

    def __call__(self, pos: torch.Tensor) -> bool:
        assert pos.shape == (3,)

        min_dist = torch.min(torch.norm(self.ref_coords - pos[None, :], dim=1))
        return min_dist <= self.distance_threshold

    def get_message_when_no_atom_found(self):
        return f"No atoms found in the reference span (distance_threshold={self.distance_threshold})."


class BallsCoordinateFilter(CoordinateFilter):
    def __init__(self, centers: torch.Tensor, radius: float):
        assert centers.ndim == 2
        assert centers.shape[0] > 0
        assert centers.shape[1] == 3
        self.centers = centers

        assert radius > 0
        self.radius = radius

    def __call__(self, pos: torch.Tensor) -> bool:
        assert pos.shape == (3,)
        min_dist = torch.min(torch.norm(self.centers - pos[None, :], dim=1))
        return min_dist <= self.radius

    def get_message_when_no_atom_found(self):
        return f"No atoms found in the balls (centers={self.centers}), radius={self.radius})."


class CompositeOrCoordinateFilter(CoordinateFilter):
    def __init__(self, coord_filters: List[CoordinateFilter]):
        assert len(coord_filters) > 0
        assert all(isinstance(f, CoordinateFilter) for f in coord_filters)
        self.coord_filters = coord_filters

    def __call__(self, pos: torch.Tensor) -> bool:
        return any(f(pos) for f in self.coord_filters)

    def get_message_when_no_atom_found(self):
        msg = "No atoms found by any of the filters:\n"
        for f in self.coord_filters:
            msg += f"- {f.get_message_when_no_atom_found()}\n"
        return msg


def _get_ligand_dict_from_sdf(
    sdf_file: Optional[Path],
    filename_depth=1,
    branch_point_coords: Optional[torch.Tensor] = None,
):
    if sdf_file is None:
        if branch_point_coords is not None:
            raise Exception("branch_point_coords must be None if sdf_file is None.")
        return {
            "element": torch.empty(
                [
                    0,
                ],
                dtype=torch.long,
            ),
            "pos": torch.empty([0, 3], dtype=torch.float),
            "atom_feature": torch.empty([0, 8], dtype=torch.float),
            "bond_index": torch.empty([2, 0], dtype=torch.long),
            "bond_type": torch.empty(
                [
                    0,
                ],
                dtype=torch.long,
            ),
        }

    mol = parse_sdf(sdf_file)
    mol = Chem.RemoveHs(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)

    n = mol.GetNumAtoms()
    assert mol.GetNumConformers() == 1
    conformer = mol.GetConformer()
    assert conformer.GetNumAtoms() == n

    bonds = list(mol.GetBonds())
    m = len(bonds)

    ligand_dict = EasyDict(
        {
            "element": torch.empty([n], dtype=torch.long),
            "pos": torch.empty([n, 3], dtype=torch.float),
            "atom_feature": torch.empty([0, 8], dtype=torch.float),
            "bond_index": torch.empty([2, 2 * m], dtype=torch.long),
            "bond_type": torch.empty([2 * m], dtype=torch.long),
        }
    )

    for i, atom in enumerate(mol.GetAtoms()):
        ligand_dict["element"][i] = atom.GetAtomicNum()
        ligand_dict["pos"][i] = torch.FloatTensor(conformer.GetAtomPosition(i))

    for j, bond in enumerate(bonds):
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        if bond_type == Chem.BondType.SINGLE:
            bond_type_int = 1
        elif bond_type == Chem.BondType.DOUBLE:
            bond_type_int = 2
        elif bond_type == Chem.BondType.TRIPLE:
            bond_type_int = 3
        else:
            raise ValueError(bond_type)

        ligand_dict["bond_index"][0, j] = begin_atom_idx
        ligand_dict["bond_index"][1, j] = end_atom_idx
        ligand_dict["bond_index"][0, m + j] = end_atom_idx
        ligand_dict["bond_index"][1, m + j] = begin_atom_idx

        ligand_dict["bond_type"][j] = bond_type_int
        ligand_dict["bond_type"][m + j] = bond_type_int

    ligand_dict["filename"] = "/".join(sdf_file.parts[-filename_depth:])

    if branch_point_coords is not None:
        assert len(branch_point_coords.shape) == 2
        assert len(branch_point_coords) > 0
        coords = ligand_dict["pos"]
        dist_matrix = torch.norm(
            coords[:, None, :] - branch_point_coords[None, :, :], dim=-1
        )
        branch_point_matrix = dist_matrix < 1e-4
        branch_point_hits = torch.sum(branch_point_matrix, dim=0)

        if not torch.all(branch_point_hits == 1):
            msg = "Branch points are invalid:\n"
            for i in range(len(branch_point_hits)):
                coord = branch_point_coords[i]
                if branch_point_hits[i] > 1:
                    msg += f"- branch point {coord} was hit {branch_point_hits[i]} times.\n"
                elif branch_point_hits[i] == 0:
                    msg += f"- branch point {coord} was not hit.\n"

            raise ValueError(msg)

        branch_point_mask = torch.any(branch_point_matrix, dim=1)
        assert torch.any(branch_point_mask)
        assert len(branch_point_mask) == len(ligand_dict["pos"])
        ligand_dict["branch_point_mask"] = branch_point_mask
        ligand_dict["focalizable_mask"] = branch_point_mask
        ligand_dict["seed_mask"] = torch.ones_like(branch_point_mask)

    return ligand_dict


def _pdb_to_pocket_data(
    pdb_path,
    coord_filter: CoordinateFilter,
    protein_filename_depth=1,
    ligand_filename_depth=1,
    ligand_sdf: Optional[Path] = None,
    model_config=None,
    branch_point_coords: Optional[torch.Tensor] = None,
):
    warnings.simplefilter("ignore", BiopythonWarning)
    ptable = Chem.GetPeriodicTable()
    parser = PDBParser()
    model = parser.get_structure(None, pdb_path)[0]

    protein_dict = EasyDict(
        {
            "element": [],
            "pos": [],
            "is_backbone": [],
            "atom_to_aa_type": [],
        }
    )
    for atom in unfold_entities(model, "A"):
        res = atom.get_parent()
        resname = res.get_resname()
        if resname == "MSE":
            resname = "MET"
        if resname not in PDBProtein.AA_NAME_NUMBER:
            continue  # Ignore water, heteros, and non-standard residues.

        element_symb = atom.element.capitalize()
        if element_symb == "H":
            continue
        x, y, z = atom.get_coord()
        pos = torch.FloatTensor([x, y, z])
        if not coord_filter(pos):
            continue

        protein_dict["element"].append(ptable.GetAtomicNumber(element_symb))
        protein_dict["pos"].append(pos)
        protein_dict["is_backbone"].append(atom.get_name() in ["N", "CA", "C", "O"])
        protein_dict["atom_to_aa_type"].append(PDBProtein.AA_NAME_NUMBER[resname])
        protein_dict["filename"] = "/".join(
            Path(pdb_path).parts[-protein_filename_depth:]
        )

    if len(protein_dict["element"]) == 0:
        raise ValueError(coord_filter.get_message_when_no_atom_found())

    protein_dict["element"] = torch.LongTensor(protein_dict["element"])
    protein_dict["pos"] = torch.stack(protein_dict["pos"], dim=0)
    protein_dict["is_backbone"] = torch.BoolTensor(protein_dict["is_backbone"])
    protein_dict["atom_to_aa_type"] = torch.LongTensor(protein_dict["atom_to_aa_type"])

    ligand_dict = _get_ligand_dict_from_sdf(
        ligand_sdf,
        filename_depth=ligand_filename_depth,
        branch_point_coords=branch_point_coords,
    )

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=protein_dict,
        ligand_dict=ligand_dict,
    )
    if ligand_sdf is None:
        mask = "all"
    else:
        mask = "none"
    data = _process_data_from_scratch(data, model_config=model_config, mask=mask)
    return data


def pdb_to_pocket_data_from_entire_pdb(pdb_path: Path, **kwargs):
    coord_filter = NoFilter()
    return _pdb_to_pocket_data(pdb_path, coord_filter, **kwargs)


def pdb_to_pocket_data_from_center(pdb_path, center, bbox_size, **kwargs):
    protein_coord_filter = BoxCoordinateFilter(center, bbox_size)
    return _pdb_to_pocket_data(pdb_path, protein_coord_filter, **kwargs)


def pdb_to_pocket_data_from_ref_span(
    pdb_path: Path,
    ref_sdf_path: Path,
    distance_threshold: float,
    **kwargs,
):
    coord_filter = RefSpanCoordinateFilter(ref_sdf_path, distance_threshold)
    return _pdb_to_pocket_data(pdb_path, coord_filter, **kwargs)


def get_pocket_data_for_opti(
    target_pdb: Path,
    seed_sdf: Path,
    branch_point_coords: torch.Tensor,
    input_range_from_seed: float,
    input_range_from_branch_points: float,
    **kwargs,
):
    f1 = RefSpanCoordinateFilter(seed_sdf, input_range_from_seed)
    f2 = BallsCoordinateFilter(branch_point_coords, input_range_from_branch_points)
    coord_filter = CompositeOrCoordinateFilter([f1, f2])
    return _pdb_to_pocket_data(
        target_pdb,
        coord_filter,
        ligand_sdf=seed_sdf,
        branch_point_coords=branch_point_coords,
        **kwargs,
    )
