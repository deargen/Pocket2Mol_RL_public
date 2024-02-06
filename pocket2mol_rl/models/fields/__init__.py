from pocket2mol_rl.models.fields.classifier import (
    SpatialClassifierVN,
    SpatialClassiferVNSkipEdge,
)


def get_field_vn(
    config, num_classes, num_edge_types, in_sca, in_vec, space_dim=3, skip_edge=False
):
    if config.name == "classifier":
        if not skip_edge:
            return SpatialClassifierVN(
                num_classes=num_classes,
                # num_indicators = num_indicators,
                num_edge_types=num_edge_types,
                in_vec=in_vec,
                in_sca=in_sca,
                num_filters=[config.num_filters, config.num_filters_vec],
                edge_channels=config.edge_channels,
                num_heads=config.num_heads,
                k=config.knn,
                cutoff=config.cutoff,
                space_dim=space_dim,
            )
        else:
            return SpatialClassiferVNSkipEdge(
                num_classes=num_classes,
                # num_indicators = num_indicators,
                num_edge_types=num_edge_types,
                in_vec=in_vec,
                in_sca=in_sca,
                num_filters=[config.num_filters, config.num_filters_vec],
                edge_channels=config.edge_channels,
                k=config.knn,
                cutoff=config.cutoff,
                space_dim=space_dim,
            )
    else:
        raise NotImplementedError("Unknown field: %s" % config.name)
