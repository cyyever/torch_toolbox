from cyy_naive_lib.log import get_logger

from .metric_logger import MetricLogger


class BatchLossLogger(MetricLogger):
    def _after_batch(self, **kwargs):
        model_executor = kwargs.get("model_executor")
        batch_size = kwargs["batch_size"]
        batch_index = kwargs["batch_index"]
        ten_batches = len(model_executor.dataset) // (10 * batch_size)
        if ten_batches == 0 or batch_index % ten_batches == 0:
            get_logger().info(
                "%s epoch: %s, batch: %s, learning rate: %s, batch loss: %s",
                self.prefix,
                kwargs["epoch"],
                batch_index,
                model_executor.get_data("cur_learning_rates", None),
                kwargs["batch_loss"],
            )
