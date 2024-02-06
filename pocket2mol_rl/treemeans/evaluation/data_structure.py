from pathlib import Path
import h5py
import numpy as np
from typing import Tuple, List, Dict, Any, Union

from pocket2mol_rl.treemeans.data.tree import PlanarTree

from dataclasses import dataclass


class TreeReconsError(Exception):
    def __init__(self, num_nodes, neighb_dict):
        self.num_nodes = num_nodes
        self.neighb_dict = neighb_dict

    def __str__(self):
        return f"Invalid tree with {self.num_nodes} nodes: {self.neighb_dict}"


def get_node_order_and_parents(num_nodes, bond_index) -> Tuple[np.ndarray, np.ndarray]:
    """
    Todo: fix key error when num_nodes = 1
    """
    assert np.all(np.logical_and(0 <= bond_index, bond_index < num_nodes))

    neighb_dict = {}
    for i, j in bond_index.T:
        neighb_dict.setdefault(i, []).append(j)
        neighb_dict.setdefault(j, []).append(i)
    for key, neighb in neighb_dict.items():
        neighb_dict[key] = sorted(list(set(neighb)))

    if num_nodes == 1:
        node_order = np.array([0])
        parents = np.array([], dtype=np.int64)
        return node_order, parents
    else:
        if any(not i in neighb_dict for i in range(num_nodes)):
            raise TreeReconsError(num_nodes, neighb_dict)

    if sum(len(neighb_dict[i]) for i in range(num_nodes) if i in neighb_dict) != 2 * (
        num_nodes - 1
    ):
        raise TreeReconsError(num_nodes, neighb_dict)

    # traverse the tree
    node_order = []
    node_to_idx = {}
    parents = []
    visited = set()
    stack = [(0, None)]
    while stack:
        node, parent = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        node_order.append(node)
        node_to_idx[node] = len(node_order) - 1
        if parent is not None:
            parents.append(node_to_idx[parent])
        stack.extend((neighb, node) for neighb in neighb_dict[node])

    # check if the tree is valid
    condition = len(node_order) == num_nodes and len(parents) == num_nodes - 1
    if not condition:
        raise TreeReconsError(num_nodes, neighb_dict)

    return np.array(node_order), np.array(parents)


def add_context(data):
    if not hasattr(data, "ligand_context_pos"):
        data.ligand_context_pos = data.ligand_pos
    if not hasattr(data, "ligand_context_element"):
        data.ligand_context_element = data.ligand_element
    if not hasattr(data, "ligand_context_bond_index"):
        data.ligand_context_bond_index = data.ligand_bond_index
    if not hasattr(data, "ligand_context_bond_type"):
        data.ligand_context_bond_type = data.ligand_bond_type
    return data


@dataclass
class SimplePlanarTree:
    vertices: np.ndarray
    parents: np.ndarray

    def __len__(self):
        return len(self.vertices)

    @classmethod
    def init_from_data(cls, data: PlanarTree, fill_context=True):
        if fill_context:
            add_context(data)
        assert hasattr(data, "ligand_context_pos")
        assert hasattr(data, "ligand_context_element")
        assert hasattr(data, "ligand_context_bond_index")
        assert hasattr(data, "ligand_context_bond_type")
        num_nodes = len(data.ligand_context_pos)
        assert data.ligand_context_pos.shape == (num_nodes, 2)
        num_bonds = len(data.ligand_context_bond_type)
        assert data.ligand_context_bond_index.shape == (2, num_bonds)
        assert data.ligand_context_bond_type.shape == (num_bonds,)

        valid_bond_idxs = (data.ligand_context_bond_type > 0).cpu().numpy()
        bond_index = data.ligand_context_bond_index[:, valid_bond_idxs].cpu().numpy()

        node_order, parents = get_node_order_and_parents(num_nodes, bond_index)
        vertices = data.ligand_context_pos[node_order, :]
        return SimplePlanarTree(vertices, parents)

    def __eq__(self, other):
        # TODO: allow for re-ordering of nodes
        return np.allclose(self.vertices, other.vertices) and np.allclose(
            self.parents, other.parents
        )
