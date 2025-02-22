import json
import os

from cyy_naive_lib.log import log_info

from ..data_pipeline.common import replace_target
from ..ml_type import DatasetType, Factory, MachineLearningPhase, TransformType
from .cache import DatasetCache
from .classification_collection import ClassificationDatasetCollection
from .collection import DatasetCollection
from .collection_sampler import (
    DatasetCollectionSplit,
    RandomSplit,
    SamplerBase,
    SplitBase,
    get_dataset_collection_sampler,
    get_dataset_collection_split,
    global_sampler_factory,
)
from .local_file import load_local_files
from .repository import DatasetFactory, get_dataset
from .sampler import DatasetSampler
from .text_collection import TextDatasetCollection
from .util import DatasetUtil

global_dataset_collection_factory: Factory = Factory()


def create_dataset_collection(
    name: str,
    dataset_kwargs: dict | None = None,
    merge_validation_to_training: bool = False,
) -> DatasetCollection:
    if dataset_kwargs is None:
        dataset_kwargs = {}
    with DatasetCache.lock:
        res = get_dataset(
            name=name,
            dataset_kwargs=dataset_kwargs,
            cache_dir=DatasetCache().get_dataset_dir(name),
        )
        if res is None:
            raise NotImplementedError(name)
        dataset_type, datasets = res
        constructor = global_dataset_collection_factory.get(dataset_type)
        if constructor is None:
            constructor = DatasetCollection
            if dataset_type == DatasetType.Text:
                constructor = TextDatasetCollection
        dc: DatasetCollection = constructor(
            datasets=datasets,
            dataset_type=dataset_type,
            name=name,
            dataset_kwargs=dataset_kwargs,
        )
        if dc.is_classification_dataset():
            dc = ClassificationDatasetCollection(dc)
        else:
            return dc
        if not merge_validation_to_training:
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
        else:
            assert not dc.has_dataset(
                MachineLearningPhase.Validation
            ) or not dc.has_dataset(MachineLearningPhase.Test)
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
        if "dataset_type" in self.dataset_kwargs:
            if isinstance(self.dataset_kwargs["dataset_type"], str):
                real_dataset_type: DatasetType | None = None
                for dataset_type in DatasetType:
                    if (
                        str(dataset_type).lower()
                        == self.dataset_kwargs["dataset_type"].lower()
                    ):
                        real_dataset_type = dataset_type
                assert real_dataset_type is not None
                self.dataset_kwargs["dataset_type"] = real_dataset_type
            assert isinstance(self.dataset_kwargs["dataset_type"], DatasetType)

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
                mode="w",
                encoding="utf-8",
            ) as f:
                json.dump(subset_indices, f)

        if self.training_dataset_indices_path is not None:
            assert subset_indices is None
            log_info(
                "use training_dataset_indices_path %s",
                self.training_dataset_indices_path,
            )
            with open(self.training_dataset_indices_path, encoding="utf-8") as f:
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
                mode="w",
                encoding="utf-8",
            ) as f:
                json.dump(label_map, f)

        if self.training_dataset_label_map_path is not None:
            assert label_map is not None
            log_info(
                "use training_dataset_label_map_path %s",
                self.training_dataset_label_map_path,
            )
            with open(self.training_dataset_label_map_path, encoding="utf-8") as f:
                self.training_dataset_label_map = json.load(f)

        if self.training_dataset_label_map is not None:
            dc.append_transform(
                transform=replace_target(self.training_dataset_label_map),
                key=TransformType.Target,
                phases=[MachineLearningPhase.Training],
            )


__all__ = [
    "DatasetSampler",
    "DatasetUtil",
    "create_dataset_collection",
    "get_dataset_collection_sampler",
    "ClassificationDatasetCollection",
    "TextDatasetCollection",
    "DatasetCollection",
    "SamplerBase",
    "SplitBase",
    "RandomSplit",
    "DatasetFactory",
    "DatasetCollectionSplit",
    "get_dataset_collection_split",
    "global_sampler_factory",
]
