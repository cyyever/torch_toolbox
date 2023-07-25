import copy
import os
import threading
from typing import Any, Callable, Generator, Iterable

import torch
from cyy_naive_lib.fs.ssd import is_ssd
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.storage import get_cached_data

from ..dataset import dataset_with_indices
from ..dataset_transform import add_data_extraction, add_transforms
from ..dataset_transform.transform import Transforms
from ..dataset_util import DatasetSplitter, get_dataset_util_cls
from ..ml_type import DatasetType, MachineLearningPhase, TransformType


class DatasetCollection:
    def __init__(
        self,
        datasets: dict[MachineLearningPhase, torch.utils.data.Dataset],
        dataset_type: DatasetType,
        name: str | None = None,
        dataset_kwargs: dict | None = None,
    ) -> None:
        self.__name: str = ""
        if name is not None:
            self.__name = name
        self.__raw_datasets: dict[
            MachineLearningPhase, torch.utils.data.Dataset
        ] = datasets
        self.__datasets: dict[MachineLearningPhase, torch.utils.data.Dataset] = {}
        for k, v in self.__raw_datasets.items():
            self.__datasets[k] = dataset_with_indices(v)
        self.__dataset_type: DatasetType = dataset_type
        self.__transforms: dict[MachineLearningPhase, Transforms] = {}
        for phase in MachineLearningPhase:
            self.__transforms[phase] = Transforms()
        if not dataset_kwargs:
            dataset_kwargs = {}
        self.__dataset_kwargs: dict = copy.deepcopy(dataset_kwargs)
        add_data_extraction(self)

    @property
    def name(self) -> str:
        return self.__name

    @property
    def dataset_kwargs(self) -> dict:
        return self.__dataset_kwargs

    def __copy__(self):
        new_obj = type(self)(
            datasets={},
            dataset_type=self.__dataset_type,
            name=self.__name,
        )
        new_obj.__raw_datasets = copy.copy(self.__raw_datasets)
        new_obj.__datasets = copy.copy(self.__datasets)
        new_obj.__transforms = copy.copy(self.__transforms)
        new_obj.__dataset_kwargs = copy.copy(self.__dataset_kwargs)
        return new_obj

    @property
    def dataset_type(self) -> DatasetType:
        return self.__dataset_type

    def foreach_raw_dataset(self) -> Generator:
        yield from self.__raw_datasets.values()

    def foreach_dataset(self) -> Generator:
        yield from self.__datasets.values()

    def has_dataset(self, phase: MachineLearningPhase) -> bool:
        return phase in self.__datasets

    def transform_dataset(
        self, phase: MachineLearningPhase, transformer: Callable
    ) -> None:
        dataset = self.get_dataset(phase)
        dataset_util = self.get_dataset_util(phase)
        self.__datasets[phase] = transformer(dataset, dataset_util, phase)

    def transform_all_datasets(self, transformer: Callable) -> None:
        for phase in self.__datasets:
            self.transform_dataset(phase, transformer)

    def set_subset(self, phase: MachineLearningPhase, indices: set) -> None:
        self.transform_dataset(
            phase=phase,
            transformer=lambda dataset, dataset_util, phase: dataset_util.get_subset(
                indices
            ),
        )

    def remove_dataset(self, phase: MachineLearningPhase) -> None:
        get_logger().debug("remove dataset %s", phase)
        self.__datasets.pop(phase, None)

    def get_dataset(self, phase: MachineLearningPhase) -> torch.utils.data.Dataset:
        return self.__datasets[phase]

    def get_transforms(self, phase: MachineLearningPhase) -> Transforms:
        return self.__transforms[phase]

    def get_dataset_util(
        self, phase: MachineLearningPhase = MachineLearningPhase.Test
    ) -> DatasetSplitter:
        return get_dataset_util_cls(dataset_type=self.dataset_type)(
            dataset=self.get_dataset(phase),
            transforms=self.__transforms[phase],
            name=self.name,
            phase=phase,
        )

    def get_original_dataset(
        self, phase: MachineLearningPhase
    ) -> torch.utils.data.Dataset:
        dataset_util = self.get_dataset_util(phase=phase)
        raw_dataset = self.__raw_datasets.get(phase)
        assert raw_dataset is not None
        dataset_util.dataset = raw_dataset
        return dataset_util.get_original_dataset()

    def append_transform(
        self, transform: Callable, key: TransformType, phases: None | Iterable = None
    ) -> None:
        for phase in MachineLearningPhase:
            if phases is not None and phase not in phases:
                continue
            self.__transforms[phase].append(key, transform)

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
    def get_dataset_dir(cls, name: str) -> str:
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
        cache_dir = os.path.join(cls.get_dataset_dir(name), ".cache")
        if phase is not None:
            cache_dir = os.path.join(cache_dir, str(phase))
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

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

    def iid_split(
        self,
        from_phase: MachineLearningPhase,
        parts: dict[MachineLearningPhase, float],
    ) -> None:
        assert self.has_dataset(phase=from_phase)
        assert parts
        get_logger().debug("split %s dataset for %s", from_phase, self.name)
        dataset_util = self.get_dataset_util(phase=from_phase)
        part_list = list(parts.items())

        datasets = dataset_util.split_by_indices(
            dataset_util.iid_split_indices([part for (_, part) in part_list])
        )
        raw_dataset = self.__raw_datasets.get(from_phase)
        assert raw_dataset is not None
        for phase, dataset in zip([phase for (phase, _) in part_list], datasets):
            self.__datasets[phase] = dataset
            self.__raw_datasets[phase] = raw_dataset

    def add_transforms(self, model_evaluator) -> None:
        add_transforms(
            dc=self,
            model_evaluator=model_evaluator,
        )

    def get_cached_data(self, file: str, computation_fun: Callable) -> Any:
        with DatasetCollection.lock:
            assert self.name is not None
            cache_dir = DatasetCollection._get_dataset_cache_dir(self.name)
            return get_cached_data(os.path.join(cache_dir, file), computation_fun)
