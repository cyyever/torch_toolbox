from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.hooks.learning_rate_hook import LearningRateHook

from .metric_visualizer import MetricVisualizer


class BatchLossLogger(MetricVisualizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__learning_rate_hook = LearningRateHook()
        self.log_times = 5

    def _after_batch(self, epoch, batch_index, batch_size, result, **kwargs):
        if self.log_times == 0:
            return
        model_executor = kwargs.get("model_executor")
        interval = model_executor.dataset_size // (self.log_times * batch_size)
        if interval == 0 or batch_index % interval == 0:
            get_logger().info(
                "%sepoch: %s, batch: %s, learning rate: %e, batch loss: %e",
                self.prefix + " " if self.prefix else "",
                epoch,
                batch_index,
                model_executor._data["cur_learning_rates"][0],
                result["loss"],
            )
