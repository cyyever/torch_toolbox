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

    # def _before_execute(self, **kwargs):
    #     model_executor = kwargs["model_executor"]
    #     if (
    #         self.__accuracy_metric is None
    #         and model_executor.model_with_loss.model_type == ModelType.Classification
    #     ):
    #         self.__accuracy_metric = AccuracyMetric()
    #     self.__exec_hook("_before_execute", **kwargs)

    # def _after_execute(self, **kwargs):
    #     self.__exec_hook("_after_execute", **kwargs)

    # def _before_batch(self, **kwargs):
    #     self.__exec_hook("_before_batch", **kwargs)

    # def _after_batch(self, **kwargs):
    #     self.__exec_hook("_after_batch", **kwargs)

    # def _before_epoch(self, **kwargs):
    #     self.__exec_hook("_before_epoch", **kwargs)

    # self.__exec_hook("_after_epoch", **kwargs)
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

    # def __exec_hook(self, hook_name: str, **kwargs):
    #     if hasattr(self.__loss_metric, hook_name):
    #         getattr(self.__loss_metric, hook_name)(**kwargs)
    #     if self.__accuracy_metric is not None:
    #         if hasattr(self.__accuracy_metric, hook_name):
    #             getattr(self.__accuracy_metric, hook_name)(**kwargs)
