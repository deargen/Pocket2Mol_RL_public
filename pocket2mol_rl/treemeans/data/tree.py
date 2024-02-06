from __future__ import annotations

import h5py
import torch

from pocket2mol_rl.data.data import ProteinLigandData
from pocket2mol_rl.treemeans.utils.graph import convert_parents_to_edge_index


class PlanarTree(ProteinLigandData):
    """Planar tree data structure for a toy task
    This code modify ProteinLigandData to fit the data structure of the toy task
    We use the same data key as ProteinLigandData to make it compatible with the existing code
    Element type for ligand and protein is equal to 0 used in this task

    Args:
        ProteinLigandData (_type_): _description_

    Raises:
        NotImplementedError: from_protein_ligand_dicts is not implemented
    """

    @staticmethod
    def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def from_hdf(hdf: h5py.File, i: int) -> PlanarTree:
        """Get data from hdf file and return a PlanarTree object

        Args:
            hdf (h5py.File): data file
            i (int): index of the data to be loaded

        Returns:
            PlanarTree:
                - ligand_element (torch.Tensor): element type of each ligand atom
                - ligand_pos (torch.Tensor): position of each ligand atom
                - ligand_bond_index (torch.Tensor): edge index of ligand atoms
                - ligand_bond_type (torch.Tensor): bond type of ligand atoms
                - ligand_nbh_list (dict): neighbor list of ligand atoms
                - protein_element (torch.Tensor): element type of each protein atom
                - protein_pos (torch.Tensor): position of each protein atom
        """
        data = PlanarTree()
        # Ligand data
        v_size = hdf["v_sizes"][i]
        data["ligand_element"] = torch.zeros(v_size, dtype=torch.long)
        data["ligand_pos"] = torch.tensor(
            hdf["v_coords"][i, :v_size, :], dtype=torch.float32
        )

        v_parents = hdf["v_parents"][i, : v_size - 1]
        data["ligand_bond_index"] = convert_parents_to_edge_index(v_parents)
        data["ligand_bond_type"] = torch.ones(
            data["ligand_bond_index"].shape[1], dtype=torch.long
        )
        data["ligand_nbh_list"] = {
            i.item(): [
                j.item()
                for k, j in enumerate(data["ligand_bond_index"][1])
                if data["ligand_bond_index"][0, k].item() == i
            ]
            for i in data["ligand_bond_index"][0]
        }

        # Protein data
        p_size = hdf["p_sizes"][i]
        data["protein_element"] = torch.zeros(p_size, dtype=torch.long)
        data["protein_pos"] = torch.tensor(
            hdf["p_coords"][i, :p_size, :], dtype=torch.float32
        )
        return data
