from typing import Any

from torchmetrics.classification import MulticlassF1Score, MultilabelF1Score

from .classification_metric import ClassificationMetric


class F1Metric(ClassificationMetric):
    __f1: None | MulticlassF1Score | MultilabelF1Score = None

    def _before_epoch(self, **kwargs) -> None:
        self.__f1 = None

    def _after_batch(self, result: dict, **kwargs: Any) -> None:
        targets = result["targets"]
        output = self._get_output(result).detach()
        if self.__f1 is None:
            executor = kwargs["executor"]
            assert executor.dataset_collection.label_number > 0
            with executor.device:
                if executor.dataset_collection.is_mutilabel():
                    self.__f1 = MultilabelF1Score(
                        num_labels=executor.dataset_collection.label_number
                    )
                else:
                    self.__f1 = MulticlassF1Score(
                        num_classes=executor.dataset_collection.label_number
                    )
        self.__f1.update(output, targets.detach())

    def _after_epoch(self, **kwargs) -> None:
        epoch = kwargs["epoch"]
        assert self.__f1 is not None
        self._set_epoch_metric(epoch, "F1", self.__f1.compute())
