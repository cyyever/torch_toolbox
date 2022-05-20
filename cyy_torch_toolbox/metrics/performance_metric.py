import os
import time

from ml_type import ModelType

from .acc_metric import AccuracyMetric
from .grad_metric import GradMetric
from .loss_metric import LossMetric
from .metric import Metric


class PerformanceMetric(Metric):
    def __init__(self, model_type, **kwargs):
        super().__init__(**kwargs)
        self.__loss_metric = LossMetric()
        self.__accuracy_metric = None
        if model_type == ModelType.Classification:
            self.__accuracy_metric = AccuracyMetric()
        self.__epoch_time_point: float = None
        self.__last_epoch = None
        self.__grad_metric = None
        if os.getenv("use_grad_norm") is not None:
            self.__grad_metric = GradMetric()

    def _before_epoch(self, **kwargs):
        self.__epoch_time_point = time.time()

    def _after_epoch(self, epoch, **kwargs):
        self.__last_epoch = epoch
        self._set_epoch_metric(epoch, "duration", time.time() - self.__epoch_time_point)
        loss_metric = self.__loss_metric.get_epoch_metric(epoch)
        for k, v in loss_metric.items():
            self._set_epoch_metric(epoch, k, v)
        if self.__accuracy_metric is not None:
            self._set_epoch_metric(
                epoch, "accuracy", self.__accuracy_metric.get_accuracy(epoch)
            )

    def get_loss(self, epoch):
        return self.get_epoch_metric(epoch, "loss")

    def get_duration(self, epoch):
        return self.get_epoch_metric(epoch, "duration")

    def get_grad_norm(self, epoch):
        if self.__grad_metric is None:
            return None
        return self.__grad_metric.get_epoch_metric(epoch, name="grad_norm")

    def get_accuracy(self, epoch):
        if self.__accuracy_metric is None:
            return None
        return self.__accuracy_metric.get_accuracy(epoch)

    def get_last_loss(self):
        return self.get_epoch_metric(self.__last_epoch, "loss")
