import torch
from typing import List


def convert_parents_to_edge_index(parents: List[int]) -> torch.Tensor:
    """Assume parent is a list where each element is the parent of the vertex represented by its `index plus one`
    For example, parents = [0, 1, 2, 1, 1, 0, 0, 4] means 1 -> 0, 2 -> 1, 3 -> 2, 4 -> 1, 5 -> 1, 6 -> 0, 7 -> 0, 8 -> 4
    return bidirectional edge index

    Args:
        parents (List[int]): parent of each vertex

    Returns:
        torch.Tensor: bidirectional edge index
    """

    # Convert parents to torch tensor
    parents = torch.tensor(parents)

    # Create an array of vertex indices
    vertices = torch.arange(len(parents)) + 1

    # Concatenate parents and vertices
    edges = torch.cat([vertices, parents], dim=0)
    edges = edges.view(2, -1)

    # Duplicate edges to get bidirectional edges
    edges = torch.cat([edges, edges.flip(0)], dim=1)
    # sort according to the first row
    edges = edges[:, edges[0, :].sort()[1]]
    return edges
