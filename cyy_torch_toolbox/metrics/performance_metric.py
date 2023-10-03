import os
import time

from cyy_torch_toolbox.ml_type import ModelType

from .acc_metric import AccuracyMetric
from .auc_metric import AUROCMetric
from .dataloader_profiler import DataloaderProfiler
from .f1_metric import F1Metric
from .grad_metric import GradMetric
from .learning_rate_metric import LearningRateMetric
from .loss_metric import LossMetric
from .metric import Metric


class PerformanceMetric(Metric):
    def __init__(self, model_type, profile: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__loss_metric = LossMetric()
        if model_type == ModelType.Classification:
            self.__accuracy_metric = AccuracyMetric()
            self.__f1_metric = F1Metric()
            self.__auc_metric = AUROCMetric()
        self.__epoch_time_point: float = time.time()
        self.__last_epoch = None
        if os.getenv("use_grad_norm") is not None:
            self.__grad_metric = GradMetric()
        if profile:
            self.__dataloader_profiler = DataloaderProfiler()
        self.__lr_metric = LearningRateMetric()

    def _before_epoch(self, **kwargs) -> None:
        self.__epoch_time_point = time.time()

    def _after_epoch(self, epoch, **kwargs) -> None:
        self.__last_epoch = epoch
        self._set_epoch_metric(epoch, "duration", time.time() - self.__epoch_time_point)

    def get_loss(self, epoch) -> float:
        return self.get_epoch_metric(epoch, "loss")

    def get_last_loss(self):
        return self.get_epoch_metric(self.__last_epoch, "loss")
