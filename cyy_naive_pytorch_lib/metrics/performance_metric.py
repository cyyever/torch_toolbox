from ml_type import ModelType

from .acc_metric import AccuracyMetric
from .loss_metric import LossMetric
from .metric import Metric


class PerformanceMetric(Metric):
    def __init__(self, model_type, **kwargs):
        super().__init__(**kwargs)
        self.__loss_metric = LossMetric()
        self.__accuracy_metric = None
        if model_type == ModelType.Classification:
            self.__accuracy_metric = AccuracyMetric()

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
