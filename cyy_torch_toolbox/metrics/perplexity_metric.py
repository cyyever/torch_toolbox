from typing import Any

from torchmetrics.text.perplexity import Perplexity

from .classification_metric import ClassificationMetric


class PerplexityMetric(ClassificationMetric):
    def _after_batch(self, result: dict, **kwargs: Any) -> None:
        executor = kwargs["executor"]
        if self._metric is None:
            self._metric = Perplexity()
        output, targets = self._get_new_output(executor, result)
        self.metric.update(output, targets.detach())

    def _after_epoch(self, **kwargs) -> None:
        epoch = kwargs["epoch"]
        self._set_epoch_metric(epoch, "perplexity", self.metric.compute())
