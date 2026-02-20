import functools

from cyy_naive_lib import Decorator
from cyy_naive_lib.log import log_info

from ..ml_type import MachineLearningPhase

labels_cache: dict[str, set[int]] = {}


class ClassificationDatasetCollection(Decorator):
    @functools.cached_property
    def label_number(self) -> int:
        return len(self.get_labels(use_cache=False))

    def get_labels(self, use_cache: bool = False) -> set[int]:
        if use_cache and self.name in labels_cache:
            return labels_cache[self.name]

        if self.name.lower() == "imagenet":
            return set(range(1000))
        labels = set()
        for phase in self.foreach_original_phase():
            labels |= self.get_original_dataset_util(phase).get_labels()
        log_info("%s label number %s", self.name, len(labels))
        labels_cache[self.name] = labels
        return labels

    def is_multilabel(self) -> bool:
        def computation_fun() -> bool:
            if self.name.lower() == "imagenet":
                return False
            for _, labels in self.get_any_dataset_util().get_batch_labels():
                if len(labels) > 1:
                    return True
            return False

        if not self.has_dataset(MachineLearningPhase.Training):
            return computation_fun()

        return self.get_cached_data("is_multilabel.pk", computation_fun)

    def get_label_names(self) -> dict[int, str]:
        def computation_fun() -> dict[int, str]:
            label_names = self.get_dataset_util(
                phase=MachineLearningPhase.Training
            ).get_label_names()
            if not label_names:
                raise NotImplementedError(f"failed to get label names for {self.name}")
            return label_names

        return computation_fun()
