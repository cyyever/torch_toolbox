from dataclasses import dataclass

import torch
import torch.cuda
from cyy_naive_lib.log import log_info

from ..metric_visualizers.performance_metric_logger import PerformanceMetricLogger
from ..metric_visualizers.performance_metric_recorder import PerformanceMetricRecorder
from ..metrics.performance_metric import PerformanceMetric
from ..ml_type import ConfigBase, MachineLearningPhase
from ..model import AMPModelEvaluator
from .cudnn import CUDNNHook
from .debugger import Debugger
from .executor_logger import ExecutorLogger
from .profiler import Profiler


@dataclass(kw_only=True)
class HookConfig(ConfigBase):
    debug: bool = False
    profile: bool = False
    use_amp: bool = False
    benchmark_cudnn: bool = True
    use_performance_metric: bool = True
    use_slow_performance_metrics: bool = False
    save_performance_metric: bool = False
    log_performance_metric: bool = True
    summarize_executor: bool = True

    def disable_log(self) -> None:
        self.log_performance_metric = False
        self.summarize_executor = False

    def set_hooks(self, executor) -> None:
        if executor.phase != MachineLearningPhase.Training:
            self.use_amp = False

        if self.use_amp:
            if not isinstance(executor.model_evaluator, AMPModelEvaluator):
                log_info("use amp")
                executor.replace_model_evaluator(AMPModelEvaluator)
        else:
            if isinstance(executor.model_evaluator, AMPModelEvaluator):
                if executor.phase == MachineLearningPhase.Training:
                    log_info("disable amp")
                executor.replace_model_evaluator(
                    lambda amp_evaluator: amp_evaluator.get_underlying_object()
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
