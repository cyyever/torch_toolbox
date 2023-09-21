from cyy_torch_toolbox.ml_type import DatasetType

from .metric import Metric


class LossMetric(Metric):
    def _after_batch(self, epoch, executor, result, batch_size, **kwargs) -> None:
        normalized_batch_loss = result["normalized_batch_loss"].clone().detach()
        epoch_loss = self.get_epoch_metric(epoch, "loss")
        if epoch_loss is None:
            epoch_loss = normalized_batch_loss
        else:
            epoch_loss += normalized_batch_loss
        self._set_epoch_metric(epoch, "loss", epoch_loss)
        if executor.dataset_collection.dataset_type == DatasetType.Text:
            self._set_epoch_metric(
                epoch, "perplexity", round(epoch_loss.exp().item(), 5)
            )

    def get_loss(self, epoch):
        return self.get_epoch_metric(epoch, "loss")
