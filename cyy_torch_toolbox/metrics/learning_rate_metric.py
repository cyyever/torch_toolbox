from .metric import Metric


class LearningRateMetric(Metric):
    def _before_batch(self, executor, batch_index, **kwargs) -> None:
        super()._before_batch(executor=executor, batch_index=batch_index, **kwargs)
        optimizer = executor.get_optimizer()
        if optimizer is not None:
            self._set_batch_metric(
                batch=batch_index,
                name="learning_rate",
                data=[group["lr"] for group in optimizer.param_groups],
            )
