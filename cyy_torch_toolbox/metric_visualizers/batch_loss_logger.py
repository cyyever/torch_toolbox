from cyy_naive_lib.log import log_info

from .metric_visualizer import MetricVisualizer


class BatchLossLogger(MetricVisualizer):
    log_times = 5

    def _after_batch(self, executor, epoch, batch_index, result, **kwargs) -> None:
        if self.log_times == 0:
            return
        if not executor.has_hook_obj("performance_metric"):
            return
        interval: int = executor.hyper_parameter.get_iterations_per_epoch(
            executor.dataset_size
        )
        if interval != 0 and batch_index % interval != 0:
            return
        performance_metric = executor.get_hook("performance_metric")
        if not performance_metric.enabled:
            return
        learning_rates = executor.get_hook("performance_metric").get_batch_metric(
            batch_index, "learning_rate"
        )
        if len(learning_rates) == 1:
            learning_rates = learning_rates[0]
        log_info(
            "%sepoch: %s, batch: %s, learning rate: %e, batch loss: %e",
            self.prefix + " " if self.prefix else "",
            epoch,
            batch_index,
            learning_rates,
            result["loss"],
        )
