import copy
import os
from collections.abc import Callable, Generator, Iterable
from typing import Any, Self

import torch
import torch.utils.data
from cyy_naive_lib.log import log_debug
from cyy_naive_lib.storage import get_cached_data

from ..data_pipeline import (
    DataPipeline,
    Transform,
    Transforms,
    append_transforms_to_dc,
    dataset_with_indices,
)
from .cache import DatasetCache
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
        assert datasets
        self.__datasets: dict[MachineLearningPhase, torch.utils.data.Dataset | list] = (
            datasets
        )
        if add_index:
            for k, v in self.__datasets.items():
                self.__datasets[k] = dataset_with_indices(v)
        self.__dataset_type: DatasetType | None = dataset_type
        self.__transforms: dict[MachineLearningPhase, Transforms] = {}
        self.__pipeline: dict[MachineLearningPhase, DataPipeline] = {}
        for phase in self.__datasets:
            self.__transforms[phase] = Transforms()
            self.__pipeline[phase] = DataPipeline()
        assert self.__transforms
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
        factor: type = global_dataset_util_factor.get(
            self.dataset_type, default=DatasetUtil
        )
        return factor(
            dataset=self.__datasets[phase],
            transforms=self.__transforms[phase],
            pipeline=self.__pipeline[phase],
            name=self.name,
        )

    def foreach_transform(self) -> Generator:
        yield from self.__transforms.items()

    def append_named_transform(
        self, transform: Transform, phases: None | Iterable = None
    ) -> None:
        for phase, pipeline in self.__pipeline.items():
            if phases is not None and phase not in phases:
                continue
            pipeline.append(transform)

    def append_transform(
        self, transform: Callable, key: TransformType, phases: None | Iterable = None
    ) -> None:
        for phase in self.__transforms:
            if phases is not None and phase not in phases:
                continue
            self.__transforms[phase].append(key, transform)

    def set_transform(
        self, transform: Callable, key: TransformType, phases: None | Iterable = None
    ) -> None:
        for phase in self.__transforms:
            if phases is not None and phase not in phases:
                continue
            self.__transforms[phase].set_one(key, transform)

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
            if phase not in self.__transforms:
                self.__transforms[phase] = copy.copy(self.__transforms[from_phase])
            if phase not in self.__pipeline:
                self.__pipeline[phase] = copy.copy(self.__pipeline[from_phase])

    def get_cached_data(self, file: str, computation_fun: Callable) -> Any:
        assert self.name is not None
        cache_dir = DatasetCache().get_dataset_cache_dir(self.name)
        return get_cached_data(os.path.join(cache_dir, file), computation_fun)
