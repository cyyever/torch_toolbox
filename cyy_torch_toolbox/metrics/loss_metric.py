from .metric import Metric


class LossMetric(Metric):
    __batch_losses: list[tuple] = []

    def _before_batch(self, **kwargs) -> None:
        self.__batch_losses = []

    def _after_batch(self, epoch, executor, result, **kwargs) -> None:
        self.__batch_losses.append(
            (result["loss"].detach().clone(), result["loss_batch_size"])
        )

    def _after_epoch(self, epoch: int, **kwargs) -> None:
        if not self.__batch_losses:
            return
        total_size = sum(item[1] for item in self.__batch_losses)
        total_loss = sum((item[1] * item[0]).item() for item in self.__batch_losses)
        self.__batch_losses = []
        self._set_epoch_metric(epoch, "loss", float(total_loss) / float(total_size))
