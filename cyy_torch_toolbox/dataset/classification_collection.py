import functools

from cyy_naive_lib.decorator import Decorator

from ..ml_type import MachineLearningPhase


class ClassificationDatasetCollection(Decorator):
    @functools.cached_property
    def label_number(self) -> int:
        return len(self.get_labels(use_cache=True))

    def get_labels(self, use_cache: bool = False) -> set:
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
        def computation_fun() -> bool:
            if self.name.lower() == "imagenet":
                return False
            for _, labels in self.get_any_dataset_util().get_batch_labels():
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

        return computation_fun()
