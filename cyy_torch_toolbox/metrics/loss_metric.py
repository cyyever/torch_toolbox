from .metric import Metric


class LossMetric(Metric):
    def _after_batch(self, **kwargs):
        epoch = kwargs.get("epoch")
        model_executor = kwargs.get("model_executor")

        real_batch_loss = kwargs.get("normalized_batch_loss").detach() / len(
            model_executor.dataset
        )
        epoch_loss = self.get_epoch_metric(epoch, "loss")
        if epoch_loss is None:
            epoch_loss = real_batch_loss
        else:
            epoch_loss += real_batch_loss
        self._set_epoch_metric(epoch, "loss", epoch_loss)

    def get_loss(self, epoch):
        return self.get_epoch_metric(epoch, "loss")
