import torch
import torch.nn.functional as F
from torch_geometric.nn.pool import knn_graph

from pocket2mol_rl.data.data import ProteinLigandData
from pocket2mol_rl.utils.tensor import concat_with_padding


class AtomComposer(object):
    def __init__(self, knn, num_edge_types=4):
        """Merge ligand context and protein to compose atoms

        Args:
            knn (_type_): knn of compose atoms
            num_edge_types (int, optional): number of edge types. Should be bigger than 1. Defaults to 4.
        """
        super().__init__()
        self.knn = knn
        assert num_edge_types > 1
        self.num_edge_types = num_edge_types

    def __call__(self, data: ProteinLigandData):
        # fetch ligand context and protein from data
        ligand_context_pos = data.ligand_context_pos
        ligand_context_feature_full = data.ligand_context_feature_full
        protein_pos = data.protein_pos
        protein_atom_feature = data.protein_atom_feature
        len_ligand_ctx = len(ligand_context_pos)
        len_protein = len(protein_pos)

        # compose ligand context and protein. save idx of them in compose
        data.compose_pos = torch.cat([ligand_context_pos, protein_pos], dim=0)
        len_compose = len_ligand_ctx + len_protein
        data.compose_feature = concat_with_padding(
            ligand_context_feature_full, protein_atom_feature
        )
        data.idx_ligand_ctx_in_compose = torch.arange(
            len_ligand_ctx, dtype=torch.long
        )  # can be delete
        data.idx_protein_in_compose = (
            torch.arange(len_protein, dtype=torch.long) + len_ligand_ctx
        )  # can be delete

        # build knn graph and bond type
        data = self.get_knn_graph(
            data, self.knn, len_ligand_ctx, len_compose, num_workers=16
        )
        return data

    def get_knn_graph(
        self,
        data: ProteinLigandData,
        knn,
        len_ligand_ctx,
        len_compose,
        num_workers=1,
    ):
        data.compose_knn_edge_index = knn_graph(
            data.compose_pos, knn, flow="target_to_source", num_workers=num_workers
        )

        id_compose_edge = (
            data.compose_knn_edge_index[0, : len_ligand_ctx * knn] * len_compose
            + data.compose_knn_edge_index[1, : len_ligand_ctx * knn]
        )
        id_ligand_ctx_edge = (
            data.ligand_context_bond_index[0] * len_compose
            + data.ligand_context_bond_index[1]
        )
        idx_edge = [torch.nonzero(id_compose_edge == id_) for id_ in id_ligand_ctx_edge]
        idx_edge = torch.tensor(
            [a.squeeze() if len(a) > 0 else torch.tensor(-1) for a in idx_edge],
            dtype=torch.long,
        )
        # for encoder edge embedding
        data.compose_knn_edge_type = torch.zeros(
            len(data.compose_knn_edge_index[0]), dtype=torch.long
        )
        data.compose_knn_edge_type[
            idx_edge[idx_edge >= 0]
        ] = data.ligand_context_bond_type[idx_edge >= 0]

        # for decoder edge feature
        data.compose_knn_edge_feature = torch.cat(
            [
                torch.ones([len(data.compose_knn_edge_index[0]), 1], dtype=torch.long),
                torch.zeros(
                    [len(data.compose_knn_edge_index[0]), self.num_edge_types - 1],
                    dtype=torch.long,
                ),
            ],
            dim=-1,
        )
        # 0 + (1,2,3)-onehot
        data.compose_knn_edge_feature[idx_edge[idx_edge >= 0]] = F.one_hot(
            data.ligand_context_bond_type[idx_edge >= 0],
            num_classes=self.num_edge_types,
        )
        return data
