from cyy_naive_lib.log import get_logger

from .metric_visualizer import MetricVisualizer


class LossMetricLogger(MetricVisualizer):
    def _after_batch(self, **kwargs):
        model_executor = kwargs.get("model_executor")
        batch_size = kwargs["batch_size"]
        batch_index = kwargs["batch_index"]
        ten_batches = len(model_executor.dataset) // (10 * batch_size)
        if ten_batches == 0 or batch_index % ten_batches == 0:
            get_logger().info(
                "epoch: %s, batch: %s, learning rate: %s, batch loss: %s",
                kwargs["epoch"],
                batch_index,
                model_executor.get_data("cur_learning_rates", None),
                kwargs["batch_loss"],
            )

    def _after_epoch(self, **kwargs):
        epoch = kwargs["epoch"]
        model_executor = kwargs.get("model_executor")
        get_logger().info(
            "epoch: %s, training loss: %s",
            epoch,
            model_executor.loss_metric.get_loss(epoch),
        )
