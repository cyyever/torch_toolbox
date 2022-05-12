from cyy_naive_lib.log import get_logger

from .metric_logger import MetricLogger


class BatchLossLogger(MetricLogger):
    def _after_batch(self, epoch, batch_index, batch_size, result, **kwargs):
        model_executor = kwargs.get("model_executor")
        five_batches = model_executor.dataset_size // (5 * batch_size)
        if five_batches == 0 or batch_index % five_batches == 0:
            get_logger().info(
                "%s epoch: %s, batch: %s, learning rate: %e, batch loss: %e",
                self.prefix,
                epoch,
                batch_index,
                model_executor.get_data("cur_learning_rates", None)[0],
                result["loss"],
            )
