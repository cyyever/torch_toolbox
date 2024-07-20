import functools
from typing import Any, Callable, Protocol

from ..ml_type import MachineLearningPhase
from .collection import DatasetCollection
from .util import DatasetUtil


class DatasetCollectionProtocol(Protocol):
    @property
    def name(self) -> str: ...

    def get_dataset_util(
        self, phase: MachineLearningPhase = MachineLearningPhase.Test
    ) -> DatasetUtil: ...

    def has_dataset(self, phase: MachineLearningPhase) -> bool: ...

    def get_cached_data(self, file: str, computation_fun: Callable) -> Any: ...


class ClassificationDatasetCollection(DatasetCollectionProtocol):

    @functools.cached_property
    def label_number(self) -> int:
        return len(self.get_labels())

    def get_labels(self, use_cache: bool = True) -> set:
        assert isinstance(self, DatasetCollection)

        def computation_fun() -> set:
            if self.name.lower() == "imagenet":
                return set(range(1000))
            labels = set()
            for phase in (
                MachineLearningPhase.Training,
                MachineLearningPhase.Validation,
                MachineLearningPhase.Test,
            ):
                if self.has_dataset(phase):
                    labels |= self.get_dataset_util(phase).get_labels()
            return labels

        if not use_cache:
            return computation_fun()

        return self.get_cached_data("labels.pk", computation_fun)

    def is_mutilabel(self) -> bool:
        assert isinstance(self, DatasetCollection)

        def computation_fun() -> bool:
            if self.name.lower() == "imagenet":
                return False
            for _, labels in self.__get_first_dataset_util().get_batch_labels():
                if len(labels) > 1:
                    return True
            return False

        if not self.has_dataset(MachineLearningPhase.Training):
            return computation_fun()

        return self.get_cached_data("is_mutilabel.pk", computation_fun)

    def get_label_names(self) -> dict:
        assert isinstance(self, DatasetCollection)

        def computation_fun():
            label_names = self.get_dataset_util(
                phase=MachineLearningPhase.Training
            ).get_label_names()
            if not label_names:
                raise NotImplementedError(f"failed to get label names for {self.name}")
            return label_names

        return self.get_cached_data("label_names.pk", computation_fun)

    def __get_first_dataset_util(self):
        assert isinstance(self, DatasetCollection)
        for phase in (
            MachineLearningPhase.Training,
            MachineLearningPhase.Validation,
            MachineLearningPhase.Test,
        ):
            if self.has_dataset(phase):
                return self.get_dataset_util(phase)
        raise RuntimeError("no dataset")
