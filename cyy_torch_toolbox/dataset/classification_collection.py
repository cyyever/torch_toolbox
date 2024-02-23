import copy
import functools
from typing import Self

from ..ml_type import MachineLearningPhase
from .collection import DatasetCollection


class ClassificationDatasetCollection:
    def __init__(self, dc: DatasetCollection):
        self.dc = dc

    def __copy__(self) -> Self:
        return type(self)(dc=copy.copy(self.dc))

    def __getattr__(self, name):
        if name == "dc":
            raise AttributeError()
        return getattr(self.dc, name)

    @functools.cached_property
    def label_number(self) -> int:
        return len(self.get_labels())

    def get_labels(self, use_cache: bool = True) -> set:
        def computation_fun() -> set:
            if self.name.lower() == "imagenet":
                return set(range(1000))
            labels = set()
            for phase in (
                MachineLearningPhase.Training,
                MachineLearningPhase.Validation,
                MachineLearningPhase.Test,
            ):
                if self.dc.has_dataset(phase):
                    labels |= self.dc.get_dataset_util(phase).get_labels()
            return labels

        if not use_cache:
            return computation_fun()

        return self.get_cached_data("labels.pk", computation_fun)

    def is_mutilabel(self) -> bool:
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
        def computation_fun():
            label_names = self.get_dataset_util(
                phase=MachineLearningPhase.Training
            ).get_label_names()
            if not label_names:
                raise NotImplementedError(f"failed to get label names for {self.name}")
            return label_names

        return self.get_cached_data("label_names.pk", computation_fun)

    def __get_first_dataset_util(self):
        for phase in (
            MachineLearningPhase.Training,
            MachineLearningPhase.Validation,
            MachineLearningPhase.Test,
        ):
            if self.dc.has_dataset(phase):
                return self.dc.get_dataset_util(phase)
        raise RuntimeError("no dataset")

    # def get_raw_data(self, phase: MachineLearningPhase, index: int) -> tuple[Any, set]:
    #     dataset_util = self.dc.get_dataset_util(phase)
    #     return (
    #         dataset_util.get_sample_raw_input(index),
    #         dataset_util.get_sample_label(index),
    #     )

    # def generate_raw_data(self, phase: MachineLearningPhase) -> Generator:
    #     dataset_util = self.dc.get_dataset_util(phase)
    #     return (
    #         self.get_raw_data(phase=phase, index=i) for i in range(len(dataset_util))
    #     )

    # @classmethod
    # def get_label(cls, label_name, label_names):
    #     reversed_label_names = {v: k for k, v in label_names.items()}
    #     return reversed_label_names[label_name]
