import copy
import json
import os
import threading
from typing import Any, Callable, Generator

import torch
from cyy_naive_lib.fs.ssd import is_ssd
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.storage import get_cached_data

from ..dataset import dataset_with_indices
from ..dataset_transform import add_data_extraction, add_transforms
from ..dataset_transform.common import replace_target
from ..dataset_transform.transform import Transforms
from ..dataset_util import (DatasetSplitter, GraphDatasetUtil, TextDatasetUtil,
                            VisionDatasetUtil)
from ..dependency import has_torchvision
from ..ml_type import DatasetType, MachineLearningPhase, TransformType
from ..tokenizer import get_tokenizer
from .dataset_repository import get_dataset

if has_torchvision:
    import torchvision


class DatasetCollection:
    def __init__(
        self,
        datasets: dict[MachineLearningPhase, torch.utils.data.Dataset],
        dataset_type: DatasetType | None = None,
        name: str | None = None,
    ) -> None:
        self.__name: str | None = name
        self.__raw_datasets: dict[
            MachineLearningPhase, torch.utils.data.Dataset
        ] = datasets
        self.__datasets: dict[MachineLearningPhase, torch.utils.data.Dataset] = {}
        for k, v in self.__raw_datasets.items():
            self.__datasets[k] = dataset_with_indices(v)
        self.__dataset_type: DatasetType | None = dataset_type
        self.__transforms: dict[MachineLearningPhase, Transforms] = {}
        for phase in MachineLearningPhase:
            self.__transforms[phase] = Transforms()
        self.tokenizer: Any | None = None

    def __copy__(self):
        new_obj = type(self)(
            datasets={},
            dataset_type=self.__dataset_type,
            name=self.__name,
        )
        new_obj.__raw_datasets = copy.copy(self.__raw_datasets)
        new_obj.__datasets = copy.copy(self.__datasets)
        new_obj.__transforms = copy.copy(self.__transforms)
        new_obj.tokenizer = copy.copy(self.tokenizer)
        return new_obj

    @property
    def dataset_type(self) -> None | DatasetType:
        return self.__dataset_type

    def transform_dataset(
        self, phase: MachineLearningPhase, transformer: Callable
    ) -> None:
        dataset = self.get_dataset(phase)
        dataset_util = self.get_dataset_util(phase)
        self.__datasets[phase] = transformer(dataset, dataset_util, phase)

    def set_subset(self, phase: MachineLearningPhase, indices: set) -> None:
        self.transform_dataset(
            phase=phase,
            transformer=lambda dataset, dataset_util, phase: dataset_util.get_subset(
                indices
            ),
        )

    def foreach_raw_dataset(self) -> Generator:
        yield from self.__raw_datasets.values()

    def foreach_dataset(self) -> Generator:
        yield from self.__datasets.values()

    def transform_all_datasets(self, transformer: Callable) -> None:
        for phase in self.__datasets:
            self.transform_dataset(phase, transformer)

    def has_dataset(self, phase: MachineLearningPhase) -> bool:
        return phase in self.__datasets

    def remove_dataset(self, phase: MachineLearningPhase) -> None:
        get_logger().debug("remove dataset %s", phase)
        self.__datasets.pop(phase, None)

    def get_dataset(self, phase: MachineLearningPhase) -> torch.utils.data.Dataset:
        return self.__datasets[phase]

    def get_transforms(self, phase: MachineLearningPhase) -> Transforms:
        return self.__transforms[phase]

    def get_original_dataset(
        self, phase: MachineLearningPhase
    ) -> torch.utils.data.Dataset:
        dataset_util = self.get_dataset_util(phase=phase)
        dataset_util.dataset = self.__raw_datasets.get(phase)
        return dataset_util.get_original_dataset()

    def get_dataset_util(
        self, phase: MachineLearningPhase = MachineLearningPhase.Test
    ) -> DatasetSplitter:
        match self.dataset_type:
            case DatasetType.Vision:
                class_name = VisionDatasetUtil
            case DatasetType.Text:
                class_name = TextDatasetUtil
            case DatasetType.Graph:
                class_name = GraphDatasetUtil
            case _:
                class_name = DatasetSplitter
        return class_name(
            dataset=self.get_dataset(phase),
            transforms=self.__transforms[phase],
            name=self.name,
            phase=phase,
        )

    def clear_transform(self, key, phases=None):
        for phase in MachineLearningPhase:
            if phases is not None and phase not in phases:
                continue
            self.__transforms[phase].clear(key)

    def append_transform(self, transform, key, phases=None):
        for phase in MachineLearningPhase:
            if phases is not None and phase not in phases:
                continue
            self.__transforms[phase].append(key, transform)

    @property
    def name(self) -> str | None:
        return self.__name

    _dataset_root_dir: str = os.path.join(os.path.expanduser("~"), "pytorch_dataset")
    lock = threading.RLock()

    @classmethod
    def get_dataset_root_dir(cls) -> str:
        with cls.lock:
            return os.getenv("pytorch_dataset_root_dir", cls._dataset_root_dir)

    @classmethod
    def set_dataset_root_dir(cls, root_dir: str) -> None:
        with cls.lock:
            cls._dataset_root_dir = root_dir

    @classmethod
    def __get_dataset_dir(cls, name: str) -> str:
        dataset_dir = os.path.join(cls.get_dataset_root_dir(), name)
        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir, exist_ok=True)
        if not is_ssd(dataset_dir):
            get_logger().warning(
                "dataset %s is not on a SSD disk: %s", name, dataset_dir
            )
        return dataset_dir

    @classmethod
    def _get_dataset_cache_dir(
        cls,
        name: str,
        phase: MachineLearningPhase | None = None,
    ) -> str:
        cache_dir = os.path.join(cls.__get_dataset_dir(name), ".cache")
        if phase is not None:
            cache_dir = os.path.join(cache_dir, str(phase))
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    @classmethod
    def create(
        cls,
        name: str,
        dataset_kwargs: dict,
    ) -> Any:
        if "root" not in dataset_kwargs:
            dataset_kwargs["root"] = cls.__get_dataset_dir(name)
        if "download" not in dataset_kwargs:
            dataset_kwargs["download"] = True
        res = get_dataset(name=name, dataset_kwargs=dataset_kwargs)
        if res is None:
            raise NotImplementedError(name)
        dataset_type, datasets = res

        dc = DatasetCollection(
            datasets=datasets,
            dataset_type=dataset_type,
            name=name,
        )
        add_data_extraction(dc)
        if dc.dataset_type == DatasetType.Text:
            dc.tokenizer = get_tokenizer(dc, dataset_kwargs.get("tokenizer", {}))
        if not dc.has_dataset(MachineLearningPhase.Validation):
            dc._split_training()
        if not dc.has_dataset(MachineLearningPhase.Test):
            dc._split_validation()
        return dc

    def is_classification_dataset(self) -> bool:
        if self.dataset_type == DatasetType.Graph:
            return True
        labels = next(
            self.get_dataset_util(phase=MachineLearningPhase.Training).get_batch_labels(
                indices=[0]
            )
        )[1]
        if len(labels) != 1:
            return False
        match next(iter(labels)):
            case int():
                return True
        return False

    def _split_training(self) -> None:
        assert (
            self.has_dataset(phase=MachineLearningPhase.Training)
            and not self.has_dataset(phase=MachineLearningPhase.Test)
            and not self.has_dataset(phase=MachineLearningPhase.Validation)
        )
        get_logger().debug("split training dataset for %s", self.name)
        dataset_util = self.get_dataset_util(phase=MachineLearningPhase.Training)
        datasets = dataset_util.decompose()
        if datasets is None:

            def computation_fun():
                return dataset_util.iid_split_indices([8, 1, 1])

            split_index_lists = computation_fun()
            datasets = dataset_util.split_by_indices(split_index_lists)
            datasets = dict(zip(MachineLearningPhase, datasets))
        raw_training_dataset = self.__raw_datasets.get(MachineLearningPhase.Training)
        for phase in (
            MachineLearningPhase.Validation,
            MachineLearningPhase.Test,
        ):
            self.__raw_datasets[phase] = raw_training_dataset
        self.__datasets = datasets

    def _split_validation(self) -> None:
        assert not self.has_dataset(
            phase=MachineLearningPhase.Test
        ) and self.has_dataset(phase=MachineLearningPhase.Validation)
        get_logger().debug("split validation dataset for %s", self.name)
        dataset_util = self.get_dataset_util(phase=MachineLearningPhase.Validation)

        def computation_fun():
            return dataset_util.iid_split_indices([1, 1])

        datasets = dataset_util.split_by_indices(computation_fun())
        self.__datasets[MachineLearningPhase.Validation] = datasets[0]
        self.__datasets[MachineLearningPhase.Test] = datasets[1]
        raw_dataset = self.__raw_datasets.get(MachineLearningPhase.Validation)
        for phase in (
            MachineLearningPhase.Validation,
            MachineLearningPhase.Test,
        ):
            self.__raw_datasets[phase] = raw_dataset

    def get_cached_data(self, file: str, computation_fun: Callable) -> Any:
        with DatasetCollection.lock:
            cache_dir = DatasetCollection._get_dataset_cache_dir(self.name)
            return get_cached_data(os.path.join(cache_dir, file), computation_fun)

    def add_transforms(self, model_evaluator, dataset_kwargs) -> None:
        add_transforms(
            dc=self, dataset_kwargs=dataset_kwargs, model_evaluator=model_evaluator
        )


class ClassificationDatasetCollection(DatasetCollection):
    @classmethod
    def create(cls, *args, **kwargs):
        dc: ClassificationDatasetCollection = DatasetCollection.create(*args, **kwargs)
        dc.__class__ = ClassificationDatasetCollection
        assert isinstance(dc, ClassificationDatasetCollection)
        return dc

    def get_labels(self, use_cache: bool = True) -> set:
        def computation_fun():
            if self.name.lower() == "imagenet":
                return range(1000)
            return self.get_dataset_util(
                phase=MachineLearningPhase.Training
            ).get_labels()

        if not use_cache:
            return computation_fun()

        return self.get_cached_data("labels.pk", computation_fun)

    def get_label_names(self) -> dict:
        def computation_fun():
            label_names = self.get_dataset_util(
                phase=MachineLearningPhase.Training
            ).get_label_names()
            if not label_names:
                raise NotImplementedError(f"failed to get label names for {self.name}")
            return label_names

        return self.get_cached_data("label_names.pk", computation_fun)

    def get_raw_data(self, phase: MachineLearningPhase, index: int) -> tuple:
        if self.dataset_type == DatasetType.Vision:
            dataset_util = self.get_dataset_util(phase)
            return (
                dataset_util.get_sample_image(index),
                dataset_util.get_sample_label(index),
            )
        if self.dataset_type == DatasetType.Text:
            dataset_util = self.get_dataset_util(phase)
            return (
                dataset_util.get_sample_text(index),
                dataset_util.get_sample_label(index),
            )
        raise NotImplementedError()

    def generate_raw_data(self, phase: MachineLearningPhase) -> Generator:
        dataset_util = self.get_dataset_util(phase)
        return (
            self.get_raw_data(phase=phase, index=i) for i in range(len(dataset_util))
        )

    @classmethod
    def get_label(cls, label_name, label_names):
        reversed_label_names = {v: k for k, v in label_names.items()}
        return reversed_label_names[label_name]

    def add_transforms(self, model_evaluator, dataset_kwargs):
        super().add_transforms(
            model_evaluator=model_evaluator, dataset_kwargs=dataset_kwargs
        )
        # add more transformers for model
        if self.dataset_type == DatasetType.Vision:
            input_size = getattr(
                model_evaluator.get_underlying_model().__class__, "input_size", None
            )
            if input_size is not None:
                get_logger().debug("resize input to %s", input_size)
                self.append_transform(
                    torchvision.transforms.Resize(input_size), key=TransformType.Input
                )
        get_logger().debug(
            "use transformers for training => \n %s",
            str(self.get_transforms(MachineLearningPhase.Training)),
        )


def create_dataset_collection(cls, name: str, dataset_kwargs: dict | None = None):
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
