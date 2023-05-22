import copy
import os
import threading
from typing import Any, Callable, Generator

import torch
from cyy_naive_lib.fs.ssd import is_ssd
from cyy_naive_lib.log import get_logger

from ..dataset import dataset_with_indices
from ..dataset_transform import add_data_extraction, add_transforms
from ..dataset_transform.transform import Transforms
from ..dataset_util import DatasetSplitter, get_dataset_util_cls
from ..ml_type import DatasetType, MachineLearningPhase
from ..tokenizer import get_tokenizer
from .dataset_repository import get_dataset


class DatasetCollection:
    def __init__(
        self,
        datasets: dict[MachineLearningPhase, torch.utils.data.Dataset],
        dataset_type: DatasetType | None = None,
        name: str | None = None,
        dataset_kwargs: dict | None = None,
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
        self.__tokenizer: Any | None = None
        if not dataset_kwargs:
            dataset_kwargs = {}
        self.__dataset_kwargs: dict = copy.deepcopy(dataset_kwargs)
        add_data_extraction(self)

    @property
    def name(self) -> str | None:
        return self.__name

    @property
    def tokenizer(self) -> Any | None:
        if self.__tokenizer is None and self.dataset_type == DatasetType.Text:
            self.__tokenizer = get_tokenizer(
                self, self.__dataset_kwargs.get("tokenizer", {})
            )
        return self.__tokenizer

    def __copy__(self):
        new_obj = type(self)(
            datasets={},
            dataset_type=self.__dataset_type,
            name=self.__name,
        )
        new_obj.__raw_datasets = copy.copy(self.__raw_datasets)
        new_obj.__datasets = copy.copy(self.__datasets)
        new_obj.__transforms = copy.copy(self.__transforms)
        new_obj.__tokenizer = copy.copy(self.__tokenizer)
        new_obj.__dataset_kwargs = copy.deepcopy(self.__dataset_kwargs)
        return new_obj

    @property
    def dataset_type(self) -> None | DatasetType:
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
        dataset_util.dataset = self.__raw_datasets.get(phase)
        return dataset_util.get_original_dataset()

    def append_transform(self, transform, key, phases=None):
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
            dataset_kwargs=dataset_kwargs,
        )
        if not dc.has_dataset(MachineLearningPhase.Validation):
            dc.__iid_split(
                from_phase=MachineLearningPhase.Training,
                parts={
                    MachineLearningPhase.Training: 8,
                    MachineLearningPhase.Validation: 1,
                    MachineLearningPhase.Test: 1,
                },
            )
        if not dc.has_dataset(MachineLearningPhase.Test):
            dc.__iid_split(
                from_phase=MachineLearningPhase.Validation,
                parts={
                    MachineLearningPhase.Validation: 1,
                    MachineLearningPhase.Test: 1,
                },
            )
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

    def __iid_split(
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
        for phase, dataset in zip([phase for (phase, _) in part_list], datasets):
            self.__datasets[phase] = dataset
            self.__raw_datasets[phase] = raw_dataset

    def add_transforms(self, model_evaluator) -> None:
        add_transforms(
            dc=self,
            dataset_kwargs=self.__dataset_kwargs,
            model_evaluator=model_evaluator,
        )
