from typing import Generator

from ..ml_type import MachineLearningPhase


class ClassificationDatasetCollection:
    # def __init__(self, dc: DatasetCollection) -> None:
    #     self.__dc = dc

    # def __getattr__(self, name: str) -> Any:
    #     return getattr(self.__dc, name)

    def get_labels(self, use_cache: bool = True) -> set:
        def computation_fun() -> set:
            if self.name is not None and self.name.lower() == "imagenet":
                return set(range(1000))
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
        dataset_util = self.get_dataset_util(phase)
        return (
            dataset_util.get_sample_raw_input(index),
            dataset_util.get_sample_label(index),
        )

    def generate_raw_data(self, phase: MachineLearningPhase) -> Generator:
        dataset_util = self.get_dataset_util(phase)
        return (
            self.get_raw_data(phase=phase, index=i) for i in range(len(dataset_util))
        )

    @classmethod
    def get_label(cls, label_name, label_names):
        reversed_label_names = {v: k for k, v in label_names.items()}
        return reversed_label_names[label_name]
