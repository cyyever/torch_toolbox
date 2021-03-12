from cyy_naive_lib.log import get_logger

from metrics.loss_metric import LossMetric
from model_executor import ModelExecutor

from .metric_visualizer import MetricVisualizer


class LossMetricLogger(MetricVisualizer):
    def __init__(self):
        super().__init__(metric=None)

    def _after_batch(self, **kwargs):
        model_executor = kwargs.get("model_executor")
        batch = kwargs["batch"]
        batch_index = kwargs["batch_index"]
        ten_batches = len(model_executor.dataset) // (
            10 * ModelExecutor.get_batch_size(batch[0])
        )
        if ten_batches == 0 or batch_index % ten_batches == 0:
            get_logger().info(
                "epoch: %s, batch: %s, learning rate: %s, batch loss: %s",
                kwargs["epoch"],
                batch_index,
                model_executor.get_data("cur_learning_rates", None),
                kwargs["batch_loss"],
            )
