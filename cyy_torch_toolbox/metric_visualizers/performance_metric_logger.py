from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import DatasetType, MachineLearningPhase

from .metric_logger import MetricLogger


class PerformanceMetricLogger(MetricLogger):
    def _after_epoch(self, **kwargs):
        epoch = kwargs["epoch"]
        model_executor = kwargs.get("model_executor")

        phase_str = "training"
        if model_executor.phase == MachineLearningPhase.Validation:
            phase_str = "validation"
        elif model_executor.phase == MachineLearningPhase.Test:
            phase_str = "test"

        metric_str = ""
        metrics = ("loss", "accuracy", "class_accuracy")
        if model_executor.dataset_collection.dataset_type == DatasetType.Text:
            metrics = metrics + ("perplexity",)
        for k in metrics:
            value = model_executor.performance_metric.get_epoch_metric(epoch, k)
            if value is not None:
                metric_str = metric_str + "{}:{}, ".format(k, value)
        metric_str = metric_str[:-2]
        get_logger().info(
            "%s epoch: %s, %s %s", self.prefix, epoch, phase_str, metric_str
        )

        get_logger().info(
            "%s epoch: %s, %s use time %s",
            self.prefix,
            epoch,
            phase_str,
            model_executor.performance_metric.get_epoch_metric(epoch, "duration"),
        )
        data_waiting_time = model_executor.performance_metric.get_epoch_metric(
            epoch, "data_waiting_time"
        )

        if data_waiting_time is not None:
            get_logger().info(
                "%s epoch: %s, %s use time %s to wait data",
                self.prefix,
                epoch,
                phase_str,
                data_waiting_time,
            )
