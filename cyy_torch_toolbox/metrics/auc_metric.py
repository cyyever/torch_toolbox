from typing import Any

from torchmetrics.classification import AUROC

from .classification_metric import ClassificationMetric


class AUROCMetric(ClassificationMetric):
    def _after_batch(self, result: dict, **kwargs: Any) -> None:
        if self._metric is None:
            executor = kwargs["executor"]
            with executor.device:
                self._metric = AUROC(
                    task=self._get_task(executor),
                    num_labels=executor.dataset_collection.label_number,
                )
        targets = result["targets"]
        output = self._get_output(result).detach()
        self.metric.update(output, targets.detach().long())

    def _after_epoch(self, **kwargs) -> None:
        epoch = kwargs["epoch"]
        self._set_epoch_metric(epoch, "AUROC", self.metric.compute())
