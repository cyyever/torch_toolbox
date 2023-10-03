from typing import Any

from torchmetrics.classification import MulticlassAUROC

from ..ml_type import ModelType
from .classification_metric import ClassificationMetric


class AUROCMetric(ClassificationMetric):
    __auroc: None | MulticlassAUROC = None

    def _before_epoch(self, **kwargs) -> None:
        executor = kwargs["executor"]
        if executor.running_model_evaluator.model_type != ModelType.Classification:
            self.__auroc = None
            return
        assert executor.dataset_collection.label_number > 0
        self.__auroc = MulticlassAUROC(
            num_classes=executor.dataset_collection.label_number
        )

    def _after_batch(self, result: dict, **kwargs: Any) -> None:
        if self.__auroc is None:
            return
        targets = result["targets"]
        self.__auroc.update(
            self._get_output(result).clone().detach().cpu(),
            targets.clone().detach().cpu(),
        )

    def _after_epoch(self, **kwargs) -> None:
        if self.__auroc is None:
            return
        epoch = kwargs["epoch"]
        self._set_epoch_metric(epoch, "AUROC", self.__auroc.compute())
