from typing import Any

from torchmetrics.classification import MulticlassAUROC

from .metric import Metric


class AUROCMetric(Metric):
    __auroc: None | MulticlassAUROC = None

    def _before_epoch(self, **kwargs) -> None:
        executor = kwargs["executor"]
        assert executor.dataset_collection.label_number > 0
        self.__auroc = MulticlassAUROC(
            num_classes=executor.dataset_collection.label_number
        )

    def _after_batch(self, result: dict, **kwargs: Any) -> None:
        assert self.__auroc is not None
        targets = result["targets"]
        self.__auroc.update(
            self._get_output(result).clone().detach().cpu(),
            targets.clone().detach().cpu(),
        )

    def _after_epoch(self, **kwargs) -> None:
        epoch = kwargs["epoch"]
        assert self.__auroc is not None
        self._set_epoch_metric(epoch, "AUROC", self.__auroc.compute())
