from cyy_naive_lib.log import get_logger

from ml_type import MachineLearningPhase

from .metric_visualizer import MetricVisualizer


class ValidationMetricLogger(MetricVisualizer):
    def __init__(self):
        super().__init__(metric=None)

    def _after_epoch(self, *args, **kwargs):
        epoch = kwargs["epoch"]
        model_executor = kwargs.get("model_executor")
        get_logger().info(
            "epoch: %s, validation loss: %s, accuracy = %s",
            epoch,
            model_executor.get_validation_metric(
                MachineLearningPhase.Validation
            ).get_epoch_metric(epoch, "loss"),
            model_executor.get_validation_metric(
                MachineLearningPhase.Validation
            ).get_epoch_metric(epoch, "accuracy"),
        )
        get_logger().info(
            "epoch: %s, test loss: %s, accuracy = %s",
            epoch,
            model_executor.get_validation_metric(
                MachineLearningPhase.Test
            ).get_epoch_metric(epoch, "loss"),
            model_executor.get_validation_metric(
                MachineLearningPhase.Test
            ).get_epoch_metric(epoch, "accuracy"),
        )
