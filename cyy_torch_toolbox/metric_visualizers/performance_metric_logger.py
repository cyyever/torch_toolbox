import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import MachineLearningPhase

from .metric_visualizer import MetricVisualizer


class PerformanceMetricLogger(MetricVisualizer):
    def _after_epoch(self, executor, epoch, **kwargs) -> None:
        phase_str = "training"
        if executor.phase == MachineLearningPhase.Validation:
            phase_str = "validation"
        elif executor.phase == MachineLearningPhase.Test:
            phase_str = "test"
        performance_metric = executor.performance_metric

        epoch_metrics = performance_metric.get_epoch_metric(epoch)
        if not epoch_metrics:
            return
        metric_str: str = ""
        for k, value in epoch_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if "accuracy" in k:
                metric_str = metric_str + "{}:{:.2%}, ".format(k, value)
            elif "loss" in k:
                metric_str = metric_str + "{}:{:.5}, ".format(k, value)
            elif k == "duration":
                metric_str = metric_str + "in {:.3} seconds, ".format(value)
            elif k == "data_waiting_time":
                metric_str = metric_str + "data loader uses {:.3} seconds, ".format(
                    value
                )
            else:
                metric_str = metric_str + f"{k}:{value}, "
        metric_str = metric_str[:-2]
        if executor.phase == MachineLearningPhase.Training:
            get_logger().info(
                "%sepoch: %s, %s %s",
                self.prefix + " " if self.prefix else "",
                epoch,
                phase_str,
                metric_str,
            )
        else:
            assert epoch == 1
            get_logger().info(
                "%s%s %s",
                self.prefix + " " if self.prefix else "",
                phase_str,
                metric_str,
            )
