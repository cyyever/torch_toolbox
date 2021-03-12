from ml_type import ModelExecutorCallbackPoint
from model_executor import ModelExecutor
from tensor import get_batch_size

from .metric import Metric


class LossMetric(Metric):
    def __init__(self, model_exetutor: ModelExecutor):
        super().__init__(model_exetutor=model_exetutor)
        self.add_callback(
            ModelExecutorCallbackPoint.AFTER_BATCH,
            self.__compute_batch_loss,
        )

    def __compute_batch_loss(self, *args, **kwargs):
        batch_loss = kwargs.get("batch_loss")
        batch = kwargs.get("batch")
        epoch = kwargs.get("epoch")
        real_batch_loss = batch_loss
        if self._model_executor.model_with_loss.is_averaged_loss():
            real_batch_loss *= get_batch_size(
                self._model_executor.decode_batch(batch)[0]
            )
        real_batch_loss /= self._model_executor.get_data("dataset_size")
        epoch_loss = self.get_epoch_metric(epoch, "loss")
        if epoch_loss is None:
            epoch_loss = real_batch_loss
        else:
            epoch_loss += real_batch_loss
        self._set_epoch_metric(epoch, "loss", epoch_loss)

    def get_loss(self, epoch):
        return self.get_epoch_metric(epoch, "loss")
