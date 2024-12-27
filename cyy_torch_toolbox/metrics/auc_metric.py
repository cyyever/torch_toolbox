from typing import Any

from torchmetrics.classification import AUROC

from .classification_metric import ClassificationMetric


class AUROCMetric(ClassificationMetric):
    def _after_batch(self, result: dict, **kwargs: Any) -> None:
        executor = kwargs["executor"]
        if self._metric is None:
            with executor.device:
                self._metric = AUROC(**self._get_metric_kwargs(executor))
        output, targets = self._get_output(executor, result)
        self.metric.update(output, targets)
        # .detach().long())

    def _after_epoch(self, **kwargs) -> None:
        epoch = kwargs["epoch"]
        self._set_epoch_metric(epoch, "AUROC", self.metric.compute())
