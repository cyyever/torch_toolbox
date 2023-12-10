from .metric import Metric


class LearningRateMetric(Metric):
    def _before_batch(self, executor, batch_index, **kwargs) -> None:
        if batch_index == 0:
            optimizer = executor.get_optimizer()
        elif executor.has_optimizer():
            optimizer = executor.get_optimizer()
        self._set_batch_metric(
            batch_index=batch_index,
            name="learning_rate",
            data=[group["lr"] for group in optimizer.param_groups],
        )
