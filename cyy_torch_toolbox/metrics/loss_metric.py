from ..ml_type import DatasetType
from .metric import Metric


class LossMetric(Metric):
    def _after_batch(self, epoch, executor, result, **kwargs) -> None:
        normalized_batch_loss = result["normalized_batch_loss"].clone().detach()
        epoch_loss = self.get_epoch_metric(epoch, "loss", to_item=False)
        if epoch_loss is None:
            epoch_loss = normalized_batch_loss
        else:
            epoch_loss += normalized_batch_loss
        self._set_epoch_metric(epoch, "loss", epoch_loss)
        if executor.dataset_collection.dataset_type == DatasetType.Text:
            self._set_epoch_metric(
                epoch, "perplexity", epoch_loss.exp().clone().detach()
            )
