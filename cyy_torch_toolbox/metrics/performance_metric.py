import time
from typing import Any

import torch

from ..ml_type import ModelType
from .acc_metric import AccuracyMetric
from .auc_metric import AUROCMetric
from .dataloader_profiler import DataloaderProfiler
from .f1_metric import F1Metric
from .grad_metric import GradMetric
from .learning_rate_metric import LearningRateMetric
from .loss_metric import LossMetric
from .metric import Metric
from .new_acc_metric import NewAccuracyMetric


class PerformanceMetric(Metric):
    def __init__(
        self,
        executor,
        profile: bool = False,
        extra_metrics: bool = False,
        use_grad_norm: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.loss_metric = LossMetric()
        if executor.running_model_evaluator.model_type in (
            ModelType.Classification,
            ModelType.TokenClassification,
        ):
            self.accuracy_metric = NewAccuracyMetric()
            # if (
            #     executor.running_model_evaluator.model_type
            #     == ModelType.TokenClassification
            # ):
            #     self.accuracy_metric = NewAccuracyMetric()
            # else:
            #     self.accuracy_metric = AccuracyMetric()
            if extra_metrics:
                self.f1_metric = F1Metric()
                self.auc_metric = AUROCMetric()
        self.__epoch_time_point: float = time.time()
        self.__last_epoch: None | int = None
        if use_grad_norm:
            self.grad_metric = GradMetric()
        if profile:
            self.dataloader_profiler = DataloaderProfiler()
        if hasattr(executor, "train"):
            self.lr_metric = LearningRateMetric()

    def _before_epoch(self, **kwargs: Any) -> None:
        self.__epoch_time_point = time.time()

    def _after_epoch(self, epoch: int, **kwargs: Any) -> None:
        self.__last_epoch = epoch
        self._set_epoch_metric(epoch, "duration", time.time() - self.__epoch_time_point)

    def get_loss(self, epoch: int, to_item: bool = True) -> float | torch.Tensor:
        return self.get_epoch_metric(epoch=epoch, name="loss", to_item=to_item)

    def get_last_loss(self) -> float | torch.Tensor:
        assert self.__last_epoch is not None
        return self.get_loss(epoch=self.__last_epoch)
