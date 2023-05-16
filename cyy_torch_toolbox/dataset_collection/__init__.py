import json
import os

from cyy_naive_lib.log import get_logger

from ..dataset_transform.common import replace_target
from ..ml_type import MachineLearningPhase, TransformType
from .classification import ClassificationDatasetCollection
from .dataset_collection import DatasetCollection


def create_dataset_collection(
    cls, name: str, dataset_kwargs: dict | None = None
) -> DatasetCollection:
    if dataset_kwargs is None:
        dataset_kwargs = {}
    with cls.lock:
        return cls.create(name=name, dataset_kwargs=dataset_kwargs)


class DatasetCollectionConfig:
    def __init__(self, dataset_name=None):
        self.dataset_name = dataset_name
        self.dataset_kwargs = {}
        self.training_dataset_percentage = None
        self.training_dataset_indices_path = None
        self.training_dataset_label_map_path = None
        self.training_dataset_label_map = None
        self.training_dataset_label_noise_percentage = None

    def create_dataset_collection(self, save_dir=None):
        if self.dataset_name is None:
            raise RuntimeError("dataset_name is None")

        dc = create_dataset_collection(
            cls=ClassificationDatasetCollection,
            name=self.dataset_name,
            dataset_kwargs=self.dataset_kwargs,
        )
        if not dc.is_classification_dataset():
            dc = create_dataset_collection(
                cls=DatasetCollection,
                name=self.dataset_name,
                dataset_kwargs=self.dataset_kwargs,
            )

        self.__transform_training_dataset(dc=dc, save_dir=save_dir)
        return dc

    def __transform_training_dataset(self, dc, save_dir=None) -> None:
        subset_indices = None
        dataset_util = dc.get_dataset_util(phase=MachineLearningPhase.Training)
        if self.training_dataset_percentage is not None:
            subset_dict = dataset_util.iid_sample(self.training_dataset_percentage)
            subset_indices = sum(subset_dict.values(), [])
            with open(
                os.path.join(save_dir, "training_dataset_indices.json"),
                mode="wt",
                encoding="utf-8",
            ) as f:
                json.dump(subset_indices, f)

        if self.training_dataset_indices_path is not None:
            assert subset_indices is None
            get_logger().info(
                "use training_dataset_indices_path %s",
                self.training_dataset_indices_path,
            )
            with open(self.training_dataset_indices_path, "r", encoding="utf-8") as f:
                subset_indices = json.load(f)
        if subset_indices is not None:
            dc.set_subset(phase=MachineLearningPhase.Training, indices=subset_indices)
        dataset_util = dc.get_dataset_util(phase=MachineLearningPhase.Training)
        label_map = None
        if self.training_dataset_label_noise_percentage:
            label_map = dataset_util.randomize_subset_label(
                self.training_dataset_label_noise_percentage
            )
            with open(
                os.path.join(
                    save_dir,
                    "training_dataset_label_map.json",
                ),
                mode="wt",
                encoding="utf-8",
            ) as f:
                json.dump(label_map, f)

        if self.training_dataset_label_map_path is not None:
            assert label_map is not None
            get_logger().info(
                "use training_dataset_label_map_path %s",
                self.training_dataset_label_map_path,
            )
            with open(self.training_dataset_label_map_path, "r", encoding="utf-8") as f:
                self.training_dataset_label_map = json.load(f)

        if self.training_dataset_label_map is not None:
            dc.append_transform(
                transform=replace_target(self.training_dataset_label_map),
                key=TransformType.Target,
                phases=[MachineLearningPhase.Training],
            )
