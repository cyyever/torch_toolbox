from cyy_naive_lib.log import get_logger

from .metric_logger import MetricLogger


class BatchLossLogger(MetricLogger):
    log_times = 5

    def _after_batch(self, epoch, batch_index, batch_size, result, **kwargs):
        if self.log_times == 0:
            return
        model_executor = kwargs.get("model_executor")
        interval = model_executor.dataset_size // (self.log_times * batch_size)
        if interval == 0 or batch_index % interval == 0:
            get_logger().info(
                "%s epoch: %s, batch: %s, learning rate: %e, batch loss: %e",
                self.prefix,
                epoch,
                batch_index,
                model_executor.get_data("cur_learning_rates", None)[0],
                result["loss"],
            )
