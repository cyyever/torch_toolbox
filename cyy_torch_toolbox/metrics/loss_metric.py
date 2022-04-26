from cyy_torch_toolbox.ml_type import DatasetType

from .metric import Metric


class LossMetric(Metric):
    def _after_batch(self, epoch, model_executor, batch_loss_sum, **kwargs):
        real_batch_loss = batch_loss_sum.detach() / len(model_executor.dataset)
        epoch_loss = self.get_epoch_metric(epoch, "loss")
        if epoch_loss is None:
            epoch_loss = real_batch_loss
        else:
            epoch_loss += real_batch_loss
        self._set_epoch_metric(epoch, "loss", epoch_loss)
        if model_executor.dataset_collection.dataset_type == DatasetType.Text:
            self._set_epoch_metric(epoch, "perplexity", epoch_loss.exp())

    def get_loss(self, epoch):
        return self.get_epoch_metric(epoch, "loss")
