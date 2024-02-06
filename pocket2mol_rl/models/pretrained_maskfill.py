import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn

from pocket2mol_rl.data.transform import FeaturizeLigandAtom, FeaturizeProteinAtom
from pocket2mol_rl.models.component import (
    AtomEmbedding,
    FrontierLayerVN,
    PositionPredictor,
    SmoothCrossEntropyLoss,
    batch_intersection_mask,
    concat_tensors_to_batch,
    embed_compose,
    get_batch_edge,
    split_tensor_by_lengths,
    split_tensor_to_segments,
)
from pocket2mol_rl.models.encoders import get_encoder_vn
from pocket2mol_rl.models.fields import get_field_vn


class PretrainedMaskFillModel(nn.Module):
    def __init__(
        self,
        config,
        ligand_featurizer: FeaturizeLigandAtom,
        protein_featurizer: FeaturizeProteinAtom,
    ):
        """Base class for MaskFillModel.
        Contains the common methods for pretraining MaskFillModel.

        Args:
            config: Config object.
            ligand_featurizer (FeaturizeLigandAtom): Ligand featurizer.
            protein_featurizer (FeaturizeProteinAtom): Protein featurizer.
        """
        super().__init__()

        num_classes = ligand_featurizer.atomic_numbers.size(0)
        self.protein_featurizer = protein_featurizer
        self.ligand_featurizer = ligand_featurizer

        self.config = config
        self.space_dim = config.space_dim
        self.num_edge_types = config.encoder.num_edge_types
        # -1, 0, 1, ..., num_edge_types-1 -> [self edge, no edge, 1, ..., num_edge_types-1]
        # This code is applied in data/transform/contrastive_sample too.
        self.edge_type_list = (
            torch.arange(self.num_edge_types + 1, dtype=torch.long) - 1
        )

        self.emb_dim = [config.hidden_channels, config.hidden_channels_vec]
        self.protein_atom_emb = AtomEmbedding(
            self.protein_featurizer.feature_dim,
            1,
            *self.emb_dim,
            space_dim=self.space_dim
        )
        self.ligand_atom_emb = AtomEmbedding(
            self.ligand_featurizer.feature_dim,
            1,
            *self.emb_dim,
            space_dim=self.space_dim
        )

        # set the hidden channels of encoder and field
        config.encoder.hidden_channels = config.hidden_channels
        config.encoder.hidden_channels_vec = config.hidden_channels_vec

        self.encoder = get_encoder_vn(config.encoder, space_dim=self.space_dim)
        in_sca, in_vec = self.encoder.out_sca, self.encoder.out_vec
        self.field = get_field_vn(
            config.field,
            num_classes=num_classes,
            num_edge_types=self.num_edge_types,
            in_sca=in_sca,
            in_vec=in_vec,
            space_dim=self.space_dim,
        )
        self.frontier_pred = FrontierLayerVN(
            in_sca=in_sca, in_vec=in_vec, hidden_dim_sca=128, hidden_dim_vec=32
        )
        self.pos_predictor = PositionPredictor(
            in_sca=in_sca,
            in_vec=in_vec,
            num_filters=[config.position.num_filters] * 2,
            n_component=config.position.n_component,
        )

        self.smooth_cross_entropy = SmoothCrossEntropyLoss(
            reduction="mean", smoothing=0.1
        )
        self.bceloss_with_logits = nn.BCEWithLogitsLoss()

    def _embed_compose(
        self,
        compose_feature,
        compose_pos,
        idx_ligand,
        idx_protein,
        compose_knn_edge_index,
        compose_knn_edge_feature,
    ):
        h_compose = embed_compose(
            compose_feature,
            compose_pos,
            idx_ligand,
            idx_protein,
            self.ligand_atom_emb,
            self.protein_atom_emb,
            self.emb_dim,
            space_dim=self.space_dim,
        )
        h_compose = self.encoder(
            node_attr=h_compose,
            pos=compose_pos,
            edge_index=compose_knn_edge_index,
            edge_feature=compose_knn_edge_feature,
        )

        return h_compose

    def get_loss(
        self,
        pos_real,
        y_real,
        pos_fake,  # query real positions,
        index_real_cps_edge_for_atten,
        tri_edge_index,
        tri_edge_feat,  # for edge attention
        edge_index_real,
        edge_label,  # edges to predict
        compose_feature,
        compose_pos,
        idx_ligand,
        idx_protein,  # compose (protein and context ligand) atoms
        y_frontier,  # frontier labels
        idx_focal,
        pos_generate,  # focal and generated positions  #NOTE: idx are in comopse
        idx_protein_all_mask,
        y_protein_frontier,  # surface of protein
        compose_knn_edge_index,
        compose_knn_edge_feature,
        real_compose_knn_edge_index,
        fake_compose_knn_edge_index,  # edges in compose, query-compose
    ):
        # # emebedding
        # (N_p+N_l, H)
        h_compose = self._embed_compose(
            compose_feature,
            compose_pos,
            idx_ligand,
            idx_protein,
            compose_knn_edge_index,
            compose_knn_edge_feature,
        )
        # # 0: frontier atoms of protein
        y_protein_frontier_pred = self.frontier_pred(h_compose, idx_protein_all_mask)
        # # 1: Fontier atoms
        y_frontier_pred = self.frontier_pred(
            h_compose,
            idx_ligand,
        )
        # # 2: Positions relative to focal atoms  `idx_focal`
        relative_pos_mu, abs_pos_mu, pos_sigma, pos_pi = self.pos_predictor(
            h_compose,
            idx_focal,
            compose_pos,
        )

        # # 3: Element and bonds of the new position atoms
        y_real_pred, edge_pred = self.field(
            pos_query=pos_real,
            edge_index_query=edge_index_real,
            pos_compose=compose_pos,
            node_attr_compose=h_compose,
            edge_index_q_cps_knn=real_compose_knn_edge_index,
            index_real_cps_edge_for_atten=index_real_cps_edge_for_atten,
            tri_edge_index=tri_edge_index,
            tri_edge_feat=tri_edge_feat,
        )  # (N_real, num_classes)

        # # fake positions
        y_fake_pred, _ = self.field(
            pos_query=pos_fake,
            edge_index_query=[],
            pos_compose=compose_pos,
            node_attr_compose=h_compose,
            edge_index_q_cps_knn=fake_compose_knn_edge_index,
        )  # (N_fake, num_classes)

        # # loss
        loss_surf = F.binary_cross_entropy_with_logits(
            input=y_protein_frontier_pred, target=y_protein_frontier.view(-1, 1).float()
        ).clamp_max(10.0)
        loss_frontier = F.binary_cross_entropy_with_logits(
            input=y_frontier_pred, target=y_frontier.view(-1, 1).float()
        ).clamp_max(10.0)
        loss_pos = (
            -torch.log(
                self.pos_predictor.get_mdn_probability(
                    abs_pos_mu, pos_sigma, pos_pi, pos_generate
                )
                + 1e-16
            )
            .mean()
            .clamp_max(10.0)
        )
        # loss_notpos = self.pos_predictor.get_mdn_probability(abs_pos_mu, pos_sigma, pos_pi, pos_notgenerate).mean()
        loss_cls = self.smooth_cross_entropy(y_real_pred, y_real.argmax(-1)).clamp_max(
            10.0
        )  # Classes
        loss_edge = F.cross_entropy(edge_pred, edge_label).clamp_max(10.0)
        # real and fake loss
        energy_real = -1 * torch.logsumexp(y_real_pred, dim=-1)  # (N_real)
        energy_fake = -1 * torch.logsumexp(y_fake_pred, dim=-1)  # (N_fake)
        energy_real = torch.clamp_max(energy_real, 40)
        energy_fake = torch.clamp_min(energy_fake, -40)
        loss_real = self.bceloss_with_logits(
            -energy_real, torch.ones_like(energy_real)
        ).clamp_max(10.0)
        loss_fake = self.bceloss_with_logits(
            -energy_fake, torch.zeros_like(energy_fake)
        ).clamp_max(10.0)

        loss = (
            torch.nan_to_num(loss_frontier)
            + torch.nan_to_num(loss_pos)
            + torch.nan_to_num(loss_cls)
            + torch.nan_to_num(loss_edge)
            + torch.nan_to_num(loss_real)
            + torch.nan_to_num(loss_fake)
            + torch.nan_to_num(loss_surf)
        )
        return (
            loss,
            loss_frontier,
            loss_pos,
            loss_cls,
            loss_edge,
            loss_real,
            loss_fake,
            torch.nan_to_num(loss_surf),
        )  # loss_notpos

    def query_batch(self, pos_query_list, batch, limit=10000):
        pos_query, batch_query = concat_tensors_to_batch(pos_query_list)
        num_query = pos_query.size(0)
        assert len(torch.unique(batch_query)) == 1, NotImplementedError(
            "Modify get_batch_edge to support multiple batches"
        )
        y_cls_all, y_ind_all = [], []
        for pos_query_partial, batch_query_partial in zip(
            split_tensor_to_segments(pos_query, limit),
            split_tensor_to_segments(batch_query, limit),
        ):
            PM = batch_intersection_mask(
                batch.protein_element_batch, batch_query_partial
            )
            LM = batch_intersection_mask(
                batch.ligand_context_element_batch, batch_query_partial
            )
            ligand_context_bond_index, ligand_context_bond_type = get_batch_edge(
                batch.ligand_context_bond_index,
                batch.ligand_context_bond_type,
            )

            y_cls_partial, y_ind_partial, _ = self(
                # Query
                pos_query=pos_query_partial,
                batch_query=batch_query_partial,
                edge_index_query=[],
                # Protein
                protein_pos=batch.protein_pos[PM],
                protein_atom_feature=batch.protein_atom_feature.float()[PM],
                batch_protein=batch.protein_element_batch[PM],
                # Ligand
                ligand_pos=batch.ligand_context_pos[LM],
                ligand_atom_feature=batch.ligand_context_feature_full.float()[LM],
                batch_ligand=batch.ligand_context_element_batch[LM],
                ligand_context_bond_index=ligand_context_bond_index,
                ligand_context_bond_type=ligand_context_bond_type,
            )
            y_cls_all.append(y_cls_partial)
            y_ind_all.append(y_ind_partial)

        y_cls_all = torch.cat(y_cls_all, dim=0)
        y_ind_all = torch.cat(y_ind_all, dim=0)

        lengths = [x.size(0) for x in pos_query_list]
        y_cls_list = split_tensor_by_lengths(y_cls_all, lengths)
        y_ind_list = split_tensor_by_lengths(y_ind_all, lengths)

        return y_cls_list, y_ind_list

    def _query_position(
        self,
        pos_query,
        h_compose,
        compose_pos,
        idx_ligand,
        ligand_bond_index,
        ligand_bond_type,
    ):
        device = pos_query.device
        # NOTE: Only one parent batch at a time (i.e. batch size = 1)
        edge_index_query = torch.stack(
            torch.meshgrid(
                torch.arange(len(pos_query), dtype=torch.int64, device=device),
                torch.arange(len(idx_ligand), dtype=torch.int64, device=device),
                indexing="ij",
            ),
            dim=0,
        ).reshape(2, -1)
        query_compose_knn_edge_index = knn(
            x=compose_pos, y=pos_query, k=self.config.field.knn, num_workers=16
        )
        (
            index_real_cps_edge_for_atten,
            tri_edge_index,
            tri_edge_feat,
        ) = self._get_tri_edges(
            edge_index_query=edge_index_query,
            pos_query=pos_query,
            idx_ligand=idx_ligand,
            ligand_bond_index=ligand_bond_index,
            ligand_bond_type=ligand_bond_type,
        )
        y_real_pred, edge_pred = self.field(
            pos_query=pos_query,
            edge_index_query=edge_index_query,
            pos_compose=compose_pos,
            node_attr_compose=h_compose,
            edge_index_q_cps_knn=query_compose_knn_edge_index,
            index_real_cps_edge_for_atten=index_real_cps_edge_for_atten,
            tri_edge_index=tri_edge_index,
            tri_edge_feat=tri_edge_feat,
        )
        edge_pred = edge_pred.reshape(
            len(pos_query), len(idx_ligand), self.num_edge_types
        )
        return y_real_pred, edge_pred

    def _get_tri_edges(
        self,
        edge_index_query,
        pos_query,
        idx_ligand,
        ligand_bond_index,
        ligand_bond_type,
        modified=False,
    ):
        row, col = edge_index_query
        device = edge_index_query.device
        acc_num_edges = 0
        index_real_cps_edge_i_list, index_real_cps_edge_j_list = (
            [],
            [],
        )  # index of real-ctx edge (for attention)
        for node in torch.arange(pos_query.size(0)):
            num_edges = (row == node).sum()
            index_edge_i = (
                torch.arange(
                    num_edges,
                    dtype=torch.long,
                ).to(device)
                + acc_num_edges
            )
            index_edge_i, index_edge_j = torch.meshgrid(
                index_edge_i, index_edge_i, indexing="ij"
            )
            index_edge_i, index_edge_j = index_edge_i.flatten(), index_edge_j.flatten()
            index_real_cps_edge_i_list.append(index_edge_i)
            index_real_cps_edge_j_list.append(index_edge_j)
            acc_num_edges += num_edges
        index_real_cps_edge_i = torch.cat(
            index_real_cps_edge_i_list, dim=0
        )  # add len(real_compose_edge_index) in the dataloader for batch
        index_real_cps_edge_j = torch.cat(index_real_cps_edge_j_list, dim=0)

        node_a_cps_tri_edge = col[
            index_real_cps_edge_i
        ]  # the node of tirangle edge for the edge attention (in the compose)
        node_b_cps_tri_edge = col[index_real_cps_edge_j]
        n_context = len(idx_ligand)
        adj_mat = (
            torch.zeros([n_context, n_context], dtype=torch.long)
            - torch.eye(n_context, dtype=torch.long)
        ).to(device)
        adj_mat[ligand_bond_index[0], ligand_bond_index[1]] = ligand_bond_type
        if modified:
            tri_edge_type = adj_mat[
                index_real_cps_edge_i, index_real_cps_edge_j
            ]  # TODO: Daeseok: Check if this is correct. If correct, make this the only option
        else:
            tri_edge_type = adj_mat[node_a_cps_tri_edge, node_b_cps_tri_edge]
        tri_edge_feat = (
            tri_edge_type.view([-1, 1]) == self.edge_type_list.to(device)
        ).long()

        index_real_cps_edge_for_atten = torch.stack(
            [
                index_real_cps_edge_i,
                index_real_cps_edge_j,  # plus len(real_compose_edge_index_0) for dataloader batch
            ],
            dim=0,
        )
        tri_edge_index = torch.stack(
            [
                node_a_cps_tri_edge,
                node_b_cps_tri_edge,  # plus len(compose_pos) for dataloader batch
            ],
            dim=0,
        )
        return index_real_cps_edge_for_atten, tri_edge_index, tri_edge_feat
