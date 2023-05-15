from ..ml_type import MachineLearningPhase
from .metric import Metric


class LearningRateMetric(Metric):
    def _before_batch(self, executor, batch_index, **kwargs):
        if executor.phase == MachineLearningPhase.Training:
            self._set_batch_metric(
                batch=batch_index,
                name="learning_rate",
                data=[group["lr"] for group in executor.get_optimizer().param_groups],
            )
