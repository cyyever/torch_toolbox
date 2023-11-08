import torch
import torch.cuda

from ..metric_visualizers.performance_metric_logger import \
    PerformanceMetricLogger
from ..metric_visualizers.performance_metric_recorder import \
    PerformanceMetricRecorder
from ..metrics.performance_metric import PerformanceMetric
from ..ml_type import MachineLearningPhase
from .amp import AMP
from .cudnn import CUDNNHook
from .debugger import Debugger
from .executor_logger import ExecutorLogger
from .profiler import Profiler


class HookConfig:
    def __init__(self) -> None:
        self.summarize_executor = True
        self.debug = False
        self.profile = False
        self.use_amp = torch.cuda.is_available()
        self.benchmark_cudnn: bool = True
        self.use_performance_metric: bool = True
        self.use_extra_performance_metrics: bool = False
        self.log_performance_metric: bool = True
        self.save_performance_metric = False

    def set_hooks(self, executor) -> None:
        if executor.phase == MachineLearningPhase.Training:
            if torch.cuda.is_available():
                executor.enable_or_disable_hook("AMP", self.use_amp, AMP())
        executor.enable_or_disable_hook("debugger", self.debug, Debugger())
        executor.enable_or_disable_hook("profiler", self.profile, Profiler())
        executor.enable_or_disable_hook(
            "logger", self.summarize_executor, ExecutorLogger()
        )
        if torch.cuda.is_available():
            executor.enable_or_disable_hook("cudnn", self.benchmark_cudnn, CUDNNHook())
        if self.use_performance_metric:
            executor.enable_hook(
                "performance_metric",
                PerformanceMetric(
                    model_type=executor.running_model_evaluator.model_type,
                    profile=self.profile,
                    extra_metrics=self.use_extra_performance_metrics,
                ),
            )
            executor.enable_or_disable_hook(
                "performance_metric_recorder",
                self.save_performance_metric,
                PerformanceMetricRecorder(),
            )
            executor.enable_or_disable_hook(
                "performance_metric_logger",
                self.log_performance_metric,
                PerformanceMetricLogger(),
            )
        else:
            executor.disable_hook("performance_metric")
            executor.disable_hook("performance_metric_recorder")
            executor.disable_hook("performance_metric_logger")
