import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import DatasetType, MachineLearningPhase

from .metric_logger import MetricLogger


class PerformanceMetricLogger(MetricLogger):
    def _after_epoch(self, model_executor, **kwargs):
        epoch = kwargs["epoch"]

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
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.item()
            if "accuracy" in k:
                metric_str = metric_str + "{}:{:.2%}, ".format(k, value)
            else:
                metric_str = metric_str + "{}:{:e}, ".format(k, value)
        metric_str = metric_str[:-2]
        get_logger().info(
            "%s epoch: %s, %s %s, in %.3f seconds",
            self.prefix,
            epoch,
            phase_str,
            metric_str,
            model_executor.performance_metric.get_epoch_metric(epoch, "duration"),
        )

        if model_executor.phase == MachineLearningPhase.Training:
            grad_norm = model_executor.performance_metric.get_grad_norm(epoch)
            if grad_norm is not None:
                get_logger().info(
                    "%s epoch: %s, grad norm is %s", self.prefix, epoch, grad_norm
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
