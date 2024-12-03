import copy
import os
import threading
from collections.abc import Callable, Generator, Iterable
from typing import Any, Self

import torch
import torch.utils.data
from cyy_naive_lib.fs.ssd import is_ssd
from cyy_naive_lib.log import log_debug, log_warning
from cyy_naive_lib.storage import get_cached_data
from cyy_naive_lib.system_info import OSType, get_operating_system_type

from ..data_pipeline import Transforms, append_transforms_to_dc, dataset_with_indices
from ..ml_type import DatasetType, MachineLearningPhase, TransformType
from .sampler import DatasetSampler
from .util import DatasetUtil, global_dataset_util_factor


class DatasetCollection:
    def __init__(
        self,
        datasets: dict[MachineLearningPhase, torch.utils.data.Dataset | list],
        dataset_type: DatasetType | None = None,
        name: str | None = None,
        dataset_kwargs: dict | None = None,
        add_index: bool = True,
    ) -> None:
        self.__name: str = "" if name is None else name
        self.__datasets: dict[MachineLearningPhase, torch.utils.data.Dataset | list] = (
            datasets
        )
        if add_index:
            for k, v in self.__datasets.items():
                self.__datasets[k] = dataset_with_indices(v)
        self.__dataset_type: DatasetType | None = dataset_type
        self.__transforms: dict[MachineLearningPhase, Transforms] = {}
        for phase in MachineLearningPhase:
            self.__transforms[phase] = Transforms()
        self.__dataset_kwargs: dict = (
            copy.deepcopy(dataset_kwargs) if dataset_kwargs else {}
        )
        append_transforms_to_dc(self)

    def __copy__(self) -> Self:
        new_obj: Self = copy.deepcopy(self)
        # pylint: disable=protected-access, unused-private-member
        new_obj.__datasets = self.__datasets.copy()
        return new_obj

    @property
    def name(self) -> str:
        return self.__name

    @property
    def dataset_kwargs(self) -> dict:
        return self.__dataset_kwargs

    @property
    def dataset_type(self) -> DatasetType:
        assert self.__dataset_type is not None
        return self.__dataset_type

    def foreach_dataset(self) -> Generator:
        yield from self.__datasets.values()

    def has_dataset(self, phase: MachineLearningPhase) -> bool:
        return phase in self.__datasets

    def add_transforms(self, model_evaluator: Any) -> None:
        append_transforms_to_dc(dc=self, model_evaluator=model_evaluator)

    def transform_dataset(
        self, phase: MachineLearningPhase, transformer: Callable
    ) -> None:
        dataset_util = self.get_dataset_util(phase)
        self.__datasets[phase] = transformer(dataset_util)

    def transform_all_datasets(self, transformer: Callable) -> None:
        for phase in self.__datasets:
            self.transform_dataset(phase, transformer)

    def set_subset(self, phase: MachineLearningPhase, indices: set) -> None:
        self.transform_dataset(
            phase=phase,
            transformer=lambda dataset_util: dataset_util.get_subset(indices),
        )

    def remove_dataset(self, phase: MachineLearningPhase) -> None:
        log_debug("remove dataset %s", phase)
        self.__datasets.pop(phase, None)

    def get_any_dataset_util(self) -> DatasetUtil:
        for phase in (
            MachineLearningPhase.Training,
            MachineLearningPhase.Validation,
            MachineLearningPhase.Test,
        ):
            if self.has_dataset(phase):
                return self.get_dataset_util(phase)
        raise RuntimeError("no dataset")

    def get_dataset_util(
        self, phase: MachineLearningPhase = MachineLearningPhase.Test
    ) -> DatasetUtil:
        factor: type = global_dataset_util_factor.get(self.dataset_type, DatasetUtil)
        return factor(
            dataset=self.__datasets[phase],
            transforms=self.__transforms[phase],
            name=self.name,
            cache_dir=self._get_dataset_cache_dir(),
        )

    def foreach_transform(self) -> Generator:
        yield from self.__transforms.items()

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
    def __get_dataset_root_dir(cls) -> str:
        with cls.lock:
            return os.getenv("PYTORCH_DATASET_ROOT_DIR", cls._dataset_root_dir)

    @classmethod
    def set_dataset_root_dir(cls, root_dir: str) -> None:
        with cls.lock:
            cls._dataset_root_dir = root_dir

    @classmethod
    def get_dataset_dir(cls, name: str) -> str:
        dataset_dir = os.path.join(cls.__get_dataset_root_dir(), name)
        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir, exist_ok=True)
        if get_operating_system_type() != OSType.Windows and not is_ssd(dataset_dir):
            log_warning("dataset %s is not on a SSD disk: %s", name, dataset_dir)
        return dataset_dir

    def _get_dataset_cache_dir(self) -> str:
        cache_dir = os.path.join(self.get_dataset_dir(self.name), ".cache")
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def is_classification_dataset(self) -> bool:
        if self.dataset_type == DatasetType.Text:
            try:
                for _, labels in self.get_dataset_util(
                    phase=MachineLearningPhase.Training
                ).get_batch_labels(indices=[0]):
                    if not labels:
                        return False
            except BaseException:
                return False
        return True

    def iid_split(
        self,
        from_phase: MachineLearningPhase,
        parts: dict[MachineLearningPhase, float],
    ) -> None:
        assert self.has_dataset(phase=from_phase)
        assert parts
        log_debug("split %s dataset for %s", from_phase, self.name)
        part_list = list(parts.items())

        sampler = DatasetSampler(dataset_util=self.get_dataset_util(phase=from_phase))
        datasets = sampler.iid_split([part for (_, part) in part_list])
        for idx, (phase, _) in enumerate(part_list):
            self.__datasets[phase] = datasets[idx]

    def get_cached_data(self, file: str, computation_fun: Callable) -> Any:
        with DatasetCollection.lock:
            assert self.name is not None
            cache_dir = self._get_dataset_cache_dir()
            return get_cached_data(os.path.join(cache_dir, file), computation_fun)
