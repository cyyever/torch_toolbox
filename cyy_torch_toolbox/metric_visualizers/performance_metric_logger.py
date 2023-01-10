import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import DatasetType, MachineLearningPhase

from .metric_visualizer import MetricVisualizer


class PerformanceMetricLogger(MetricVisualizer):
    def _after_epoch(self, model_executor, epoch, **kwargs):
        phase_str = "training"
        if model_executor.phase == MachineLearningPhase.Validation:
            phase_str = "validation"
        elif model_executor.phase == MachineLearningPhase.Test:
            phase_str = "test"
        performance_metric = model_executor.performance_metric

        epoch_metrics = performance_metric.get_epoch_metrics(epoch)
        if not epoch_metrics:
            return
        metric_str: str = ""
        for k, value in epoch_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if "accuracy" in k:
                metric_str = metric_str + "{}:{:.2%}, ".format(k, value)
            elif k == "duration":
                metric_str = metric_str + "in {:.3} seconds, ".format(value)
            elif k == "data_waiting_time":
                metric_str = metric_str + "data loader uses {:.3} seconds, ".format(
                    value
                )
            else:
                metric_str = metric_str + f"{k}:{value}, "
        metric_str = metric_str[:-2]
        get_logger().info(
            "%sepoch: %s, %s %s",
            self.prefix + " " if self.prefix else "",
            epoch,
            phase_str,
            metric_str,
        )
