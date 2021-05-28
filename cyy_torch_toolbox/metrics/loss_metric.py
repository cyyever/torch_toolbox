from .metric import Metric


class LossMetric(Metric):
    def _after_batch(self, **kwargs):
        batch_loss = kwargs.get("batch_loss")
        epoch = kwargs.get("epoch")
        batch_size = kwargs.get("batch_size")
        model_executor = kwargs.get("model_executor")
        real_batch_loss = batch_loss
        if model_executor.model_with_loss.is_averaged_loss():
            real_batch_loss *= batch_size
        real_batch_loss /= len(model_executor.dataset)
        epoch_loss = self.get_epoch_metric(epoch, "loss")
        if epoch_loss is None:
            epoch_loss = real_batch_loss
        else:
            epoch_loss += real_batch_loss
        self._set_epoch_metric(epoch, "loss", epoch_loss)

    def get_loss(self, epoch):
        return self.get_epoch_metric(epoch, "loss")
