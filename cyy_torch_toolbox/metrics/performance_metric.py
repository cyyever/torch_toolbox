import os
import time

from cyy_torch_toolbox.ml_type import ModelType

from .acc_metric import AccuracyMetric
from .dataloader_profiler import DataloaderProfiler
from .grad_metric import GradMetric
from .loss_metric import LossMetric
from .metric import Metric


class PerformanceMetric(Metric):
    def __init__(self, model_type, profile: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.__loss_metric = LossMetric()
        self.__accuracy_metric = None
        if model_type == ModelType.Classification:
            self.__accuracy_metric = AccuracyMetric()
        self.__epoch_time_point: float = time.time()
        self.__last_epoch = None
        if os.getenv("use_grad_norm") is not None:
            self.__grad_metric = GradMetric()
        if profile:
            self.__dataloader_profiler = DataloaderProfiler()

    def _before_epoch(self, **kwargs):
        self.__epoch_time_point = time.time()

    def _after_epoch(self, epoch, **kwargs):
        self.__last_epoch = epoch
        self._set_epoch_metric(epoch, "duration", time.time() - self.__epoch_time_point)

    def get_loss(self, epoch):
        return self.__loss_metric.get_epoch_metric(epoch, "loss")

    def get_accuracy(self, epoch):
        if self.__accuracy_metric is None:
            return None
        return self.__accuracy_metric.get_accuracy(epoch)

    def get_last_loss(self):
        return self.get_epoch_metric(self.__last_epoch, "loss")
