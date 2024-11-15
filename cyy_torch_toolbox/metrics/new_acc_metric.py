from typing import Any

import torch
from torchmetrics.classification import Accuracy

from .metric import Metric


class NewAccuracyMetric(Metric):
    __acc: None | Accuracy = None

    def _before_execute(self, **kwargs: Any) -> None:
        self.__acc = None

    @torch.no_grad()
    def _after_batch(self, result: dict, **kwargs: Any) -> None:
        executor = kwargs["executor"]
        if self.__acc is None:
            with executor.device:
                if executor.dataset_collection.label_number <= 2:
                    self.__acc = Accuracy(task="binary")
                else:
                    self.__acc = Accuracy(
                        task="multiclass",
                        num_classes=executor.dataset_collection.label_number,
                    )

        targets = result["targets"]
        output = result.get("model_output")
        if output is None:
            output = result.get("logits")
        assert isinstance(output, torch.Tensor)
        assert isinstance(targets, torch.Tensor)
        mask = targets != -100
        new_output = output[mask]

        assert self.__acc is not None
        if executor.dataset_collection.label_number <= 2 and new_output.shape[-1]==2:
            new_output=torch.argmax(new_output, dim=-1)
        self.__acc.update(new_output, targets[mask].detach())

    def _after_epoch(self, **kwargs) -> None:
        epoch = kwargs["epoch"]
        assert self.__acc is not None
        self._set_epoch_metric(epoch, "accuracy", self.__acc.compute())
