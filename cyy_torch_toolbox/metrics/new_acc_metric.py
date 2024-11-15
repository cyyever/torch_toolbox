from typing import Any

import torch
import torchmetrics.metric
from torchmetrics.classification import Accuracy

from .classification_metric import ClassificationMetric


class NewAccuracyMetric(ClassificationMetric):
    __acc: None | torchmetrics.metric.Metric = None

    def _before_execute(self, **kwargs: Any) -> None:
        self.__acc = None

    @torch.no_grad()
    def _after_batch(self, result: dict, **kwargs: Any) -> None:
        executor = kwargs["executor"]
        if self.__acc is None:
            with executor.device:
                self.__acc = Accuracy(
                    task=self._get_task(executor),
                    num_classes=executor.dataset_collection.label_number,
                )
        output, targets = self._get_new_output(executor, result)
        self.__acc.update(output, targets)

    def _after_epoch(self, **kwargs) -> None:
        epoch = kwargs["epoch"]
        assert self.__acc is not None
        self._set_epoch_metric(epoch, "accuracy", self.__acc.compute())
