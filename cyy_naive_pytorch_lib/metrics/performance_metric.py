from ml_type import MachineLearningPhase, ModelType

from .acc_metric import AccuracyMetric
from .loss_metric import LossMetric
from .metric import Metric


class PerformanceMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__loss_metric = LossMetric()
        self.__accuracy_metric = None
        # self.__phase = phase
        # self.__inferencer = None

    def _before_execute(self, **kwargs):
        model_executor = kwargs.get("model_executor")
        self.__loss_metric.prepend_to_model_executor_before_other_callback(
            model_executor, self
        )
        if model_executor.model_with_loss.model_type == ModelType.Classification:
            self.__accuracy_metric = AccuracyMetric()
            self.__accuracy_metric.prepend_to_model_executor_before_other_callback(
                model_executor, self
            )
        else:
            self.__accuracy_metric = None
        # if model_executor.phase != MachineLearningPhase.Training:
        #     self.__inferencer = model_executor.get_inferencer(
        #         self.__phase, copy_model=False
        #     )
        #     self.__inferencer.remove_logger()

    def _after_epoch(self, **kwargs):
        epoch = kwargs.get("epoch")
        self._set_epoch_metric(epoch, "loss", self.__loss_metric.get_loss(epoch))
        self._set_epoch_metric(
            epoch, "accuracy", self.__accuracy_metric.get_accuracy(epoch)
        )
