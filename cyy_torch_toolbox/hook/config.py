import copy
from typing import Any, Self

import torch
import torch.cuda
from cyy_naive_lib.log import log_info

from ..metric_visualizers.performance_metric_logger import \
    PerformanceMetricLogger
from ..metric_visualizers.performance_metric_recorder import \
    PerformanceMetricRecorder
from ..metrics.performance_metric import PerformanceMetric
from ..ml_type import MachineLearningPhase
from ..model import AMPModelEvaluator
from .cudnn import CUDNNHook
from .debugger import Debugger
from .executor_logger import ExecutorLogger
from .profiler import Profiler


class HookConfig:
    def __init__(self) -> None:
        self.summarize_executor = True
        self.debug = False
        self.profile = False
        self.use_amp = False
        self.benchmark_cudnn: bool = True
        self.use_performance_metric: bool = True
        self.use_slow_performance_metrics: bool = False
        self.log_performance_metric: bool = True
        self.save_performance_metric = False
        self.__old_config: Any = None

    def disable_log(self) -> None:
        self.summarize_executor = False
        self.use_performance_metric = False

    def __enter__(self) -> Self:
        self.__old_config = copy.copy(self)
        return self

    def __exit__(self, *args, **kwargs) -> None:
        for name in dir(self):
            if not name.startswith("_"):
                setattr(self, name, getattr(self.__old_config, name))
        self.__old_config = None

    def set_hooks(self, executor) -> None:
        if executor.phase != MachineLearningPhase.Training:
            self.use_amp = False
        if executor.device.type.lower() == "mps":
            self.use_amp = False

        if self.use_amp:
            if not isinstance(executor.model_evaluator, AMPModelEvaluator):
                log_info("use amp")
                executor.replace_model_evaluator(AMPModelEvaluator)
        else:
            if isinstance(executor.model_evaluator, AMPModelEvaluator):
                log_info("disable amp")
                executor.replace_model_evaluator(
                    lambda amp_evaluator: amp_evaluator.evaluator
                )

        executor.append_or_disable_hook("debugger", self.debug, Debugger())
        executor.append_or_disable_hook("profiler", self.profile, Profiler())
        executor.append_or_disable_hook(
            "logger", self.summarize_executor, ExecutorLogger()
        )
        if torch.cuda.is_available():
            executor.append_or_disable_hook("cudnn", self.benchmark_cudnn, CUDNNHook())
        executor.append_or_disable_hook(
            "performance_metric",
            self.use_performance_metric,
            PerformanceMetric(
                executor=executor,
                profile=self.profile,
                extra_metrics=self.use_slow_performance_metrics,
            ),
        )
        if self.use_performance_metric:
            executor.append_or_disable_hook(
                "performance_metric_recorder",
                self.save_performance_metric,
                PerformanceMetricRecorder(),
            )
            executor.append_or_disable_hook(
                "performance_metric_logger",
                self.log_performance_metric,
                PerformanceMetricLogger(),
            )
        else:
            executor.disable_hook("performance_metric")
            executor.disable_hook("performance_metric_recorder")
            executor.disable_hook("performance_metric_logger")
