from .metric import Metric


class LearningRateMetric(Metric):
    def _before_batch(self, executor, batch_index, **kwargs) -> None:
        optimizer = executor.get_optimizer()
        assert optimizer is not None
        self._set_batch_metric(
            batch_index=batch_index,
            name="learning_rate",
            data=[group["lr"] for group in optimizer.param_groups],
        )
