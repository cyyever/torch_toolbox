from typing import Any

import torch
from torchmetrics.classification import Accuracy

from .classification_metric import ClassificationMetric


class NewAccuracyMetric(ClassificationMetric):
    @torch.no_grad()
    def _after_batch(self, result: dict, **kwargs: Any) -> None:
        executor = kwargs["executor"]
        if self._metric is None:
            with executor.device:
                self._metric = Accuracy(**self._get_metric_kwargs(executor))
        output, targets = self._get_output(executor, result)
        if output.numel() > 0:
            self.metric.update(output, targets)

    def _after_epoch(self, **kwargs: Any) -> None:
        epoch = kwargs["epoch"]
        self._set_epoch_metric(epoch, "accuracy", self.metric.compute())
