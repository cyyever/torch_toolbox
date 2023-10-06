from typing import Any
import torch

from torchmetrics.classification import MulticlassF1Score, MultilabelF1Score

from .classification_metric import ClassificationMetric


class F1Metric(ClassificationMetric):
    __f1: None | MulticlassF1Score | MultilabelF1Score = None

    def _before_epoch(self, **kwargs) -> None:
        executor = kwargs["executor"]
        assert executor.dataset_collection.label_number > 0
        if executor.dataset_collection.is_mutilabel():
            self.__f1 = MultilabelF1Score(
                num_labels=executor.dataset_collection.label_number
            )
        else:
            self.__f1 = MulticlassF1Score(
                num_classes=executor.dataset_collection.label_number
            )

    def _after_batch(self, result: dict, **kwargs: Any) -> None:
        targets = result["targets"]
        output = self._get_output(result).clone().detach().cpu().view(-1)
        if len(output.shape) == 1:
            output = torch.stack((1 - output, output), dim=1)
        assert self.__f1 is not None
        self.__f1.update(
            output,
            targets.clone().detach().cpu(),
        )

    def _after_epoch(self, **kwargs) -> None:
        epoch = kwargs["epoch"]
        assert self.__f1 is not None
        self._set_epoch_metric(epoch, "F1", self.__f1.compute())
