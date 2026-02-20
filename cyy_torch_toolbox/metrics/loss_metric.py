from typing import Any

import torch

from .metric import Metric


class LossMetric(Metric):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.__batch_losses: list[tuple] = []

    def _before_epoch(self, **kwargs: Any) -> None:
        self.__batch_losses = []

    def _after_batch(self, result, **kwargs: Any) -> None:
        loss_batch_size = result["loss_batch_size"]
        if isinstance(loss_batch_size, torch.Tensor):
            loss_batch_size = loss_batch_size.detach().to(
                device="cpu", non_blocking=True
            )
        self.__batch_losses.append(
            (
                result["loss"].detach().to(device="cpu", non_blocking=True),
                loss_batch_size,
            )
        )

    def _after_epoch(self, epoch: int, **kwargs: Any) -> None:
        if not self.__batch_losses:
            return
        total_size = sum(
            (item[1] for item in self.__batch_losses[1:]),
            start=self.__batch_losses[0][1],
        )
        total_loss = sum(
            (item[1] * item[0] for item in self.__batch_losses[1:]),
            start=self.__batch_losses[0][1] * self.__batch_losses[0][0],
        )
        self.__batch_losses = []
        self._set_epoch_metric(epoch, "loss", total_loss / total_size)
