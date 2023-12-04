import json
import os
from typing import Type

from cyy_naive_lib.log import get_logger

from ..data_pipeline.common import replace_target
from ..ml_type import DatasetType, MachineLearningPhase, TransformType
from .classification import ClassificationDatasetCollection
from .dataset_collection import DatasetCollection
from .dataset_repository import get_dataset
from .text import TextDatasetCollection


def create_dataset_collection(
    name: str, dataset_kwargs: dict | None = None
) -> DatasetCollection:
    if dataset_kwargs is None:
        dataset_kwargs = {}
    with DatasetCollection.lock:
        if "root" not in dataset_kwargs:
            dataset_kwargs["root"] = DatasetCollection.get_dataset_dir(name)
        if "download" not in dataset_kwargs:
            dataset_kwargs["download"] = True
        res = get_dataset(name=name, dataset_kwargs=dataset_kwargs)
        if res is None:
            raise NotImplementedError(name)
        dataset_type, datasets = res

        cls: Type = DatasetCollection
        if dataset_type == DatasetType.Text:
            cls = TextDatasetCollection
        dc: DatasetCollection = cls(
            datasets=datasets,
            dataset_type=dataset_type,
            name=name,
            dataset_kwargs=dataset_kwargs,
        )
        if dc.is_classification_dataset():
            cls = dc.__class__
            dc.__class__ = cls.__class__(
                cls.__name__ + "WithClassification",
                (cls, ClassificationDatasetCollection),
                {},
            )

        if not dc.has_dataset(MachineLearningPhase.Validation):
            dc.iid_split(
                from_phase=MachineLearningPhase.Training,
                parts={
                    MachineLearningPhase.Training: 8,
                    MachineLearningPhase.Validation: 1,
                    MachineLearningPhase.Test: 1,
                },
            )
        if not dc.has_dataset(MachineLearningPhase.Test):
            dc.iid_split(
                from_phase=MachineLearningPhase.Validation,
                parts={
                    MachineLearningPhase.Validation: 1,
                    MachineLearningPhase.Test: 1,
                },
            )
        return dc


class DatasetCollectionConfig:
    def __init__(self, dataset_name: str = "") -> None:
        self.dataset_name: str = dataset_name
        self.dataset_kwargs: dict = {}
        self.training_dataset_percentage = None
        self.training_dataset_indices_path = None
        self.training_dataset_label_map_path = None
        self.training_dataset_label_map = None
        self.training_dataset_label_noise_percentage = None

    def create_dataset_collection(
        self, save_dir: str | None = None
    ) -> DatasetCollection:
        assert self.dataset_name is not None
        dc = create_dataset_collection(
            name=self.dataset_name, dataset_kwargs=self.dataset_kwargs
        )

        self.__transform_training_dataset(dc=dc, save_dir=save_dir)
        return dc

    def __transform_training_dataset(self, dc, save_dir: str | None = None) -> None:
        subset_indices = None
        dataset_util = dc.get_dataset_util(phase=MachineLearningPhase.Training)
        if self.training_dataset_percentage is not None:
            subset_dict = dataset_util.iid_sample(self.training_dataset_percentage)
            subset_indices = sum(subset_dict.values(), [])
            assert save_dir is not None
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
            assert save_dir is not None
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
