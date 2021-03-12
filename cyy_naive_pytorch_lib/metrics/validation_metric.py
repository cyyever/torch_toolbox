from ml_type import MachineLearningPhase

from .metric import Metric


class ValidationMetric(Metric):
    def __init__(self, phase: MachineLearningPhase, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert phase != MachineLearningPhase.Training
        self.__phase = phase

    def _after_epoch(self, **kwargs):
        model_executor = kwargs.get("model_executor")
        epoch = kwargs.get("epoch")
        inferencer = model_executor.get_inferencer(self.__phase, copy_model=False)
        inferencer.inference()
        self._set_epoch_metric(epoch, "loss", inferencer.loss_metric.get_loss(1))
        self._set_epoch_metric(epoch, "accuracy", inferencer.accuracy_metric.get_accuracy(1))
