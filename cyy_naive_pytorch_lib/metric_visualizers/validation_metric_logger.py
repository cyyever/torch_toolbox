from cyy_naive_lib.log import get_logger
from ml_type import MachineLearningPhase

from .metric_visualizer import MetricVisualizer


class ValidationMetricLogger(MetricVisualizer):
    def __init__(self):
        super().__init__(metric=None)

    def _after_epoch(self, *args, **kwargs):
        epoch = kwargs["epoch"]
        model_executor = kwargs.get("model_executor")
        for phase in [MachineLearningPhase.Validation, MachineLearningPhase.Test]:
            phase_str = "validation"
            if phase == MachineLearningPhase.Test:
                phase_str = "test"
            get_logger().info(
                "epoch: %s, %s loss: %s, accuracy = %s",
                epoch,
                phase_str,
                model_executor.get_validation_metric(phase).get_epoch_metric(
                    epoch, "loss"
                ),
                model_executor.get_validation_metric(phase).get_epoch_metric(
                    epoch, "accuracy"
                ),
            )
            get_logger().info(
                "epoch: %s, %s class accuracy = %s",
                epoch,
                phase_str,
                model_executor.get_validation_metric(phase).get_epoch_metric(
                    epoch, "class_accuracy"
                ),
            )
