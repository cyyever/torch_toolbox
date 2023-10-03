from typing import Any

from torchmetrics.classification import MulticlassAUROC, MultilabelAUROC

from .classification_metric import ClassificationMetric


class AUROCMetric(ClassificationMetric):
    __auroc: None | MulticlassAUROC | MultilabelAUROC = None

    def _before_epoch(self, **kwargs) -> None:
        executor = kwargs["executor"]
        assert executor.dataset_collection.label_number > 0
        if executor.dataset_collection.is_mutilabel():
            self.__auroc = MultilabelAUROC(
                num_labels=executor.dataset_collection.label_number
            )
        else:
            self.__auroc = MulticlassAUROC(
                num_classes=executor.dataset_collection.label_number
            )

    def _after_batch(self, result: dict, **kwargs: Any) -> None:
        targets = result["original_targets"]
        self.__auroc.update(
            self._get_output(result).clone().detach().cpu(),
            targets.clone().detach().cpu().long(),
        )

    def _after_epoch(self, **kwargs) -> None:
        epoch = kwargs["epoch"]
        self._set_epoch_metric(epoch, "AUROC", self.__auroc.compute())
