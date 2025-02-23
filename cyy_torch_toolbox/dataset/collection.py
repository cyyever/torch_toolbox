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
    DatasetWithIndex,
    Transform,
    append_transforms_to_dc,
)
from ..ml_type import DatasetType, MachineLearningPhase
from .cache import DatasetCache
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
        if add_index:
            for k, v in datasets.items():
                v_index = DatasetWithIndex()(v)
                v_index.original_dataset = v
                datasets[k] = v_index
        self.__datasets: dict[MachineLearningPhase, torch.utils.data.Dataset | list] = (
            datasets
        )
        self.__dataset_type: DatasetType | None = dataset_type
        self.__pipeline: dict[MachineLearningPhase, DataPipeline] = {}
        for phase in self.__datasets:
            self.__pipeline[phase] = DataPipeline()
        assert self.__pipeline
        self.__dataset_kwargs: dict = (
            copy.deepcopy(dataset_kwargs) if dataset_kwargs else {}
        )
        self.has_enhanced_data_pipeline: bool = False

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

    def clear_pipelines(self) -> None:
        for p in self.__pipeline.values():
            p.clear()

    def foreach_dataset(self) -> Generator:
        yield from self.__datasets.values()

    def has_dataset(self, phase: MachineLearningPhase) -> bool:
        return phase in self.__datasets

    def add_data_pipeline(self, model_evaluator: Any) -> None:
        if not self.has_enhanced_data_pipeline:
            append_transforms_to_dc(dc=self, model_evaluator=model_evaluator)
            self.has_enhanced_data_pipeline = True

    def transform_dataset(
        self,
        phase: MachineLearningPhase,
        transformer: Callable[[DatasetUtil], torch.utils.data.Dataset],
    ) -> None:
        dataset_util = self.get_dataset_util(phase)
        self.__datasets[phase] = transformer(dataset_util)

    def transform_all_datasets(
        self, transformer: Callable[[DatasetUtil], torch.utils.data.Dataset]
    ) -> None:
        for phase in self.__datasets:
            self.transform_dataset(phase, transformer)

    def set_subset(self, phase: MachineLearningPhase, indices: set[int]) -> None:
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
            pipeline=self.__pipeline[phase],
            name=self.name,
        )

    def prepend_named_transform(
        self, transform: Transform, phases: None | Iterable = None
    ) -> None:
        for phase, pipeline in self.__pipeline.items():
            if phases is not None and phase not in phases:
                continue
            pipeline.prepend(transform)

    def append_named_transform(
        self, transform: Transform, phases: None | Iterable = None
    ) -> None:
        for phase, pipeline in self.__pipeline.items():
            if phases is not None and phase not in phases:
                continue
            pipeline.append(transform)

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
        for (phase, _), dataset in zip(part_list, datasets, strict=False):
            self.__datasets[phase] = dataset
            if phase not in self.__pipeline:
                self.__pipeline[phase] = copy.deepcopy(self.__pipeline[from_phase])

    def get_cached_data(self, file: str, computation_fun: Callable) -> Any:
        assert self.name is not None
        cache_dir = DatasetCache().get_dataset_cache_dir(self.name)
        return get_cached_data(os.path.join(cache_dir, file), computation_fun)
