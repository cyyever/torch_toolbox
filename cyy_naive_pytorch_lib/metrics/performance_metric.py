from ml_type import ModelType

from .acc_metric import AccuracyMetric
from .loss_metric import LossMetric
from .metric import Metric


class PerformanceMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__loss_metric = LossMetric()
        self.__accuracy_metric = None

    def append_to_model_executor(self, model_executor):
        self.__loss_metric.append_to_model_executor(model_executor)
        if model_executor.model_with_loss.model_type == ModelType.Classification:
            self.__accuracy_metric = AccuracyMetric()
            self.__accuracy_metric.append_to_model_executor(model_executor)
        else:
            self.__accuracy_metric = None
        super().append_to_model_executor(model_executor)

    def _after_epoch(self, **kwargs):
        epoch = kwargs.get("epoch")
        self._set_epoch_metric(epoch, "loss", self.__loss_metric.get_loss(epoch))
        if self.__accuracy_metric is not None:
            self._set_epoch_metric(
                epoch, "accuracy", self.__accuracy_metric.get_accuracy(epoch)
            )

    def get_loss(self, epoch):
        return self.get_epoch_metric(epoch, "loss")

    def get_accuracy(self, epoch):
        if self.__accuracy_metric is None:
            return None
        return self.get_epoch_metric(epoch, "accuracy")
