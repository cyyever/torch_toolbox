from cyy_naive_lib.log import get_logger
from model_util import ModelUtil

from .metric_visualizer import MetricVisualizer


class LossMetricLogger(MetricVisualizer):
    def __init__(self):
        super().__init__(metric=None)

    def _before_execute(self, **kwargs):
        trainer = kwargs["model_executor"]
        model_util = ModelUtil(trainer.model)
        get_logger().info("dataset size is %s", len(trainer.dataset))
        get_logger().info("use device %s", trainer.device)
        get_logger().info(
            "hyper_parameter is %s, optimizer is %s, lr_scheduler is %s, %s, parameter number is %s",
            trainer.hyper_parameter,
            trainer.get_optimizer(),
            trainer.get_lr_scheduler(),
            trainer.model_with_loss,
            len(model_util.get_parameter_list()),
        )

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
