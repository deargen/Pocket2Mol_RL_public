import torch
from torch_geometric.data import Batch

from pocket2mol_rl.data.data import ProteinLigandData

from pocket2mol_rl.rl.model.actor import (
    MolgenActor,
    Episode,
    SuccessfulEpisode,
    FailedEpisode,
    FailedCompleteEpisode,
    FailedIntermediateEpisode,
)

from math import ceil
from tqdm import tqdm
from pathlib import Path


from typing import Any, Optional, List, Union

from pocket2mol_rl.utils.mol import RdkitMol, write_sdf

from pocket2mol_rl.utils.silence import silence_rdkit

silence_rdkit()

from typing import Generator

from abc import ABCMeta, abstractmethod

from rdkit import Chem

from scipy.spatial.distance import cdist
import numpy as np

from pocket2mol_rl.evaluation.quick_evaluate import (
    _get_local_geometry_vals,
    _get_flatness_vals,
)


class MoleculeFilter(metaclass=ABCMeta):
    @abstractmethod
    def __call__(
        self, e: SuccessfulEpisode, mol: RdkitMol, data: ProteinLigandData
    ) -> bool:
        pass


class MinGenerationLengthFilter(MoleculeFilter):
    def __init__(self, generation_min_length: int):
        self.generation_min_length = generation_min_length

    def __call__(
        self, e: SuccessfulEpisode, mol: RdkitMol, data: ProteinLigandData
    ) -> bool:
        generation_length = len(e.actions) - 1
        return generation_length >= self.generation_min_length


class StereoProblemsFilter(MoleculeFilter):
    def __init__(self):
        pass

    def __call__(
        self, e: SuccessfulEpisode, mol: RdkitMol, data: ProteinLigandData
    ) -> bool:
        num_bonds, num_valid_bonds, num_angles, num_valid_angles, _, _, num_clashes = (
            _get_local_geometry_vals(mol)
        )

        if any(
            x is None
            for x in [
                num_bonds,
                num_valid_bonds,
                num_angles,
                num_valid_angles,
                num_clashes,
            ]
        ):
            return False
        if num_valid_bonds != num_bonds:
            return False
        if num_valid_angles != num_angles:
            return False
        if num_clashes > 0:
            return False

        num_rings, num_valid_rings = _get_flatness_vals(mol)
        if any(x is None for x in [num_rings, num_valid_rings]):
            return False
        if num_valid_rings != num_rings:
            return False

        return True


class CompositeAndFilter(MoleculeFilter):
    def __init__(self, filters: List[MoleculeFilter]):
        self.filters = filters

    def __call__(
        self, e: SuccessfulEpisode, mol: RdkitMol, data: ProteinLigandData
    ) -> bool:
        return all(f(e, mol, data) for f in self.filters)


def get_molecule_filter(**kwargs) -> CompositeAndFilter:
    filters = []

    if (
        "generation_min_length" in kwargs
        and kwargs["generation_min_length"] is not None
    ):
        generation_min_length = kwargs["generation_min_length"]
        filters.append(MinGenerationLengthFilter(generation_min_length))

    if "filter_stereo_problems" in kwargs and kwargs["filter_stereo_problems"]:
        filters.append(StereoProblemsFilter())

    return CompositeAndFilter(filters)


def _get_bond_map(mol: RdkitMol):
    bonds = mol.GetBonds()
    bond_map = {}
    for bond in bonds:
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        if begin_idx > end_idx:
            begin_idx, end_idx = end_idx, begin_idx
        bond_map[(begin_idx, end_idx)] = bond_type
    return bond_map


def is_novel(
    mols: List[RdkitMol],
    mol: RdkitMol,
    unique_distance_threshold=0.3,
    unique_smiles=False,
) -> bool:
    if len(mols) == 0:
        return True

    # find potential matches
    potential_matches = []
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    for mol2 in mols:
        smiles2 = Chem.MolToSmiles(mol2, isomericSmiles=True, canonical=True)
        if smiles == smiles2:
            potential_matches.append(smiles2)

    if len(potential_matches) == 0:
        return True

    if unique_smiles:
        return False

    # Compare one-to-one correspondence of atoms
    coords = mol.GetConformer().GetPositions()
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    bond_map = _get_bond_map(mol)
    for mol2 in mols:
        coords2 = mol2.GetConformer().GetPositions()
        elements2 = [atom.GetSymbol() for atom in mol2.GetAtoms()]

        dist_matrix = cdist(coords, coords2)
        dist_match = dist_matrix < unique_distance_threshold

        if not np.all(np.sum(dist_match, axis=0) == 1):
            continue  # not a match
        if not np.all(np.sum(dist_match, axis=1) == 1):
            continue  # not a match

        to_continue = False
        mol_to_mol2_idx_map = {i: j for i, j in zip(*np.where(dist_match))}
        for i, j in mol_to_mol2_idx_map.items():
            if elements[i] != elements2[j]:
                to_continue = True  # not a match
                break
        if to_continue:
            continue

        to_continue = False
        bond_map2 = _get_bond_map(mol2)
        for i1, i2 in bond_map.keys():
            j1 = mol_to_mol2_idx_map[i1]
            j2 = mol_to_mol2_idx_map[i2]
            if j1 > j2:
                j1, j2 = j2, j1
            if (j1, j2) not in bond_map2:
                to_continue = True  # not a match
                break
            if bond_map[(i1, i2)] != bond_map2[(j1, j2)]:
                to_continue = True  # not a match
                break
        if to_continue:
            continue

        return False  # match

    return True  # no match


def generate_exact_number(
    actor: MolgenActor,
    initial_data,
    n: int,
    max_batch_size: int = 16,
    batch_factor: float = 1.1,
    return_mol: bool = True,
    return_data_object: bool = False,
    sdf_dir: Optional[Path] = None,
    pos_only_mean=True,
    hard_stop=0.5,
    unique=False,
    unique_smiles=False,
    unique_distance_threshold=0.3,
    batch_tolerance=4,
    generator=None,
    **filter_kwargs,
) -> Generator[Union[RdkitMol, Path], None, None]:
    """_summary_

    Args:
        actor (MolgenActor):
        initial_data (ProteinLigandData):
        n (int): _description_
        device (str, optional):
        max_batch_size(int, optional):
        batch_factor (float, optional):
        return_mol (bool, optional):
        sdf_dir (Optional[Path], optional):

    Returns:
        generated (Union[List[RdkitMol], List[Path]]): generated mol objects of sdf file paths, depending on return_mol.
    """

    if not return_mol:
        assert sdf_dir is not None
    if sdf_dir is not None:
        sdf_dir = Path(sdf_dir)
        sdf_files = list(sdf_dir.glob("*.sdf"))
        if len(sdf_files) > 0:
            raise ValueError(
                f"Directory {sdf_dir} is already filled with sdf files. Please empty it or choose another directory."
            )
        sdf_dir.mkdir(parents=True, exist_ok=True)

    assert n > 0, n

    print("Generating molecules..")
    pbar = tqdm(total=n)
    molecule_filter = get_molecule_filter(**filter_kwargs)
    generated_mols = []
    num_generated = 0
    num_tolerated = 0
    batch_idx = 0
    while num_generated < n:
        batch_idx += 1
        batch_size = min(max_batch_size, ceil(batch_factor * (n - num_generated)))
        batch = [initial_data.clone() for _ in range(batch_size)]
        episodes = actor.get_rollouts(
            batch, pos_only_mean=pos_only_mean, hard_stop=hard_stop, generator=generator
        )
        successful_episodes = [e for e in episodes if isinstance(e, SuccessfulEpisode)]

        batch_num_generated = 0
        for i, e in enumerate(successful_episodes):
            mol = e.obj
            data = e.final_data
            if not molecule_filter(e, mol, data):
                continue

            if unique or unique_smiles:
                if not is_novel(
                    generated_mols,
                    mol,
                    unique_distance_threshold=unique_distance_threshold,
                    unique_smiles=unique_smiles,
                ):
                    continue

            if sdf_dir is not None:
                idx = num_generated + 1
                sdf_file = sdf_dir / f"{idx}.sdf"
                write_sdf(mol, sdf_file)
            if return_data_object:
                output = e.final_data
            elif return_mol:
                output = mol
            else:
                assert sdf_dir is not None
                output = sdf_file
            yield output

            batch_num_generated += 1
            num_generated += 1
            generated_mols.append(mol)
            pbar.update(1)

            if num_generated == n:
                break

        if batch_num_generated == 0:
            num_tolerated += 1
            if num_tolerated >= batch_tolerance:
                break
        else:
            num_tolerated = 0

        pbar.set_postfix(
            {
                "batch": batch_idx,
                "num": num_generated,
                "tol": f"{num_tolerated}/{batch_tolerance}",
            }
        )

    pbar.close()

    if num_tolerated < batch_tolerance:
        assert num_generated == n
