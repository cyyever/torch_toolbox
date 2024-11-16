from typing import Any

from torchmetrics.classification import F1Score

from .classification_metric import ClassificationMetric


class F1Metric(ClassificationMetric):
    def _after_batch(self, result: dict, **kwargs: Any) -> None:
        executor = kwargs["executor"]
        if self._metric is None:
            assert executor.dataset_collection.label_number > 0
            with executor.device:
                self._metric = F1Score(**self._get_metric_kwargs(executor))
        output, targets = self._get_new_output(executor, result)
        self.metric.update(output, targets.detach())

    def _after_epoch(self, **kwargs) -> None:
        epoch = kwargs["epoch"]
        self._set_epoch_metric(epoch, "F1", self.metric.compute())
