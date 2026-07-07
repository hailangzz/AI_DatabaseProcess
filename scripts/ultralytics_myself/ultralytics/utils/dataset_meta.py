from dataclasses import dataclass
from pathlib import Path
import yaml

import torch


@dataclass
class DatasetInfo:
    """
    Information of a single dataset.
    """

    id: int

    name: str

    weight: float

    nc: int

    train: str

    val: str

    names: dict

    class_map: dict

    class_mask: torch.Tensor

    def to_data_dict(self):
        return {
            "train": self.train,
            "val": self.val,
            "nc": self.nc,
            "names": self.names,
            "channels": 3,
        }


class DatasetMeta:
    """
    Manage all dataset information for cross-dataset training.

    Example

    meta.global_nc

    meta.datasets[0]

    meta.get_class_mask(0)

    meta.local_to_global(0, 0)
    """

    def __init__(self, yaml_file):

        self.yaml_file = str(yaml_file)

        self.global_nc = 0

        self.global_names = {}

        self.dataset_num = 0

        self.cross_dataset = False

        self.datasets = {}

        self._load_yaml()

    ##########################################################################
    # Load yaml
    ##########################################################################

    def _load_yaml(self):

        with open(self.yaml_file, "r") as f:
            cfg = yaml.safe_load(f)

        self.global_nc = cfg["nc"]

        self.global_names = cfg["names"]

        self.cross_dataset = cfg.get("cross_dataset", False)

        multi_datasets = cfg["multi_datasets"]

        self.dataset_num = len(multi_datasets)

        for item in multi_datasets:

            dataset = DatasetInfo(

                id=item["id"],

                name=item["name"],

                weight=float(item.get("weight", 1.0)),

                nc=item["nc"],

                train=item["train"],

                val=item["val"],

                names=item["data"]["names"],

                class_map=item["data"]["class_map"],

                class_mask=torch.tensor(
                    item["data"]["class_mask"],
                    dtype=torch.float32,
                ),
            )

            self.datasets[dataset.id] = dataset

    ##########################################################################
    # APIs
    ##########################################################################

    def get_dataset(self, dataset_id):

        return self.datasets[dataset_id]

    def get_class_mask(self, dataset_id):

        return self.datasets[dataset_id].class_mask

    def get_class_map(self, dataset_id):

        return self.datasets[dataset_id].class_map

    def get_weight(self, dataset_id):

        return self.datasets[dataset_id].weight

    def get_names(self, dataset_id):

        return self.datasets[dataset_id].names

    ##########################################################################
    # Class Mapping
    ##########################################################################

    def local_to_global(self, dataset_id, local_cls):

        mapper = self.datasets[dataset_id].class_map

        return mapper[int(local_cls)]

    def global_to_local(self, dataset_id, global_cls):

        mapper = self.datasets[dataset_id].class_map

        inverse = {v: k for k, v in mapper.items()}

        return inverse.get(global_cls, -1)

    ##########################################################################
    # Utilities
    ##########################################################################

    def get_all_masks(self):

        masks = []

        for i in sorted(self.datasets.keys()):

            masks.append(self.datasets[i].class_mask)

        return torch.stack(masks, dim=0)

    def __repr__(self):

        text = []

        text.append("=" * 60)

        text.append("DatasetMeta")

        text.append("=" * 60)

        text.append(f"Global Classes : {self.global_nc}")

        text.append(f"Datasets       : {self.dataset_num}")

        text.append(f"Cross Dataset  : {self.cross_dataset}")

        text.append("")

        for dataset in self.datasets.values():

            text.append(f"[{dataset.id}] {dataset.name}")

            text.append(f" weight     : {dataset.weight}")

            text.append(f" nc         : {dataset.nc}")

            text.append(f" class_map  : {dataset.class_map}")

            text.append(f" class_mask : {dataset.class_mask.tolist()}")

            text.append("")

        return "\n".join(text)