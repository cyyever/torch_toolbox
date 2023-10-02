from typing import Any

from torchmetrics.classification import MulticlassF1Score

from .metric import Metric


class F1Metric(Metric):
    __f1_score_counter: None | MulticlassF1Score = None

    def _before_epoch(self, **kwargs) -> None:
        executor = kwargs["executor"]
        assert executor.dataset_collection.label_number > 0
        self.__f1_score_counter = MulticlassF1Score(
            num_classes=executor.dataset_collection.label_number
        )

    def _after_batch(self, result: dict, **kwargs: Any) -> None:
        output = result["model_output"]
        logits = result.get("logits", None)
        targets = result["targets"]
        if logits is not None:
            output = logits
        self.__f1_score_counter.update(
            output.clone().detach().cpu(), targets.clone().detach().cpu()
        )

    def _after_epoch(self, **kwargs) -> None:
        epoch = kwargs["epoch"]
        self._set_epoch_metric(epoch, "F1", self.__f1_score_counter.compute())
