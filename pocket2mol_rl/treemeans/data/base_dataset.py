import h5py
import numpy as np
from torch.utils.data import Dataset, Subset

from pocket2mol_rl import replace_root_dir
from pocket2mol_rl.treemeans.data.tree import PlanarTree
import numpy as np


def get_dataset(config, *args, **kwargs):
    """Get dataset. If path is starts with ".", it will be replaced with ROOT_DIR."""
    hdf_path = replace_root_dir(config.path)
    dataset = BaseDataset(hdf_path, *args, **kwargs)
    subsets = dataset.get_subsets(config)
    return dataset, subsets


class BaseDataset(Dataset):
    def __init__(self, hdf_path: str, transform=None):
        """Implement custom dataloader from this by implementing
        - Custom __getitem__(self, i)
        - Custom DataLoader sampler (e.g. for train/val/test split)
        - Custom collate_fn
        Load data from hdf file and return a PlanarTree object

        Todo:
            - Test batch iterator

        Args:
            hdf_path (str): file path to the hdf file
            transform (_type_, optional): Transform to be applied to the data. Defaults to None.
        """
        self.hdf = h5py.File(hdf_path, "r")
        self.total_num_samples = len(self.hdf["v_sizes"])
        self.max_num_vertices = self.hdf["v_coords"].shape[1]
        self.max_num_points = self.hdf["p_coords"].shape[1]
        self.transform = transform

    def get_subsets(self, config):
        iso_classes = self.hdf["tree_iso_classes"][:]
        unique_iso_classes = np.unique(iso_classes)
        np.random.seed(config.seed)
        np.random.shuffle(iso_classes)
        num_train_iso_classes = int(len(unique_iso_classes) * config.train_ratio)
        num_val_iso_classes = int(len(unique_iso_classes) * config.val_ratio)
        num_test_iso_classes = (
            len(unique_iso_classes) - num_train_iso_classes - num_val_iso_classes
        )

        train_iso_classes = unique_iso_classes[:num_train_iso_classes]
        val_iso_classes = unique_iso_classes[
            num_train_iso_classes : num_train_iso_classes + num_val_iso_classes
        ]
        test_iso_classes = unique_iso_classes[
            num_train_iso_classes + num_val_iso_classes :
        ]
        iso_classes_dict = {
            "train": train_iso_classes,
            "val": val_iso_classes,
            "test": test_iso_classes,
        }

        idxs_dict = {}
        for mode, mode_iso_classes in iso_classes_dict.items():
            mode_idxs = []
            for iso_class in mode_iso_classes:
                idxs = np.where(iso_classes == iso_class)[0]
                assert len(idxs) >= 1
                if len(idxs) > config.max_datapoints_per_iso_class:
                    np.random.seed(config.seed)
                    idxs = np.random.choice(
                        idxs, size=config.max_datapoints_per_iso_class, replace=False
                    )
                    assert len(idxs) == config.max_datapoints_per_iso_class
                elif len(idxs) < config.max_datapoints_per_iso_class:
                    if config.repeat_iso_class_if_fewer:
                        q, r = divmod(config.max_datapoints_per_iso_class, len(idxs))
                        np.random.seed(config.seed)
                        idxs = np.concatenate(
                            [
                                np.repeat(idxs, q),
                                np.random.choice(idxs, size=r, replace=False),
                            ],
                            axis=0,
                        )
                        assert len(idxs) == config.max_datapoints_per_iso_class
                mode_idxs.extend(idxs.tolist())
            mode_idxs.sort()
            idxs_dict[mode] = mode_idxs

        return {k: Subset(self, indices=v) for k, v in idxs_dict.items()}

    def __len__(self):
        return self.total_num_samples

    def __getitem__(self, i) -> PlanarTree:
        """
        To include model-specific preprocessing,
        override this method with call to super().__getitem__(i)
        """
        data = PlanarTree.from_hdf(self.hdf, i)
        data.id = i
        if self.transform is not None:
            data = self.transform(data)
        return data
