import torch
import torch.cuda

from ..metric_visualizers.performance_metric_recorder import \
    PerformanceMetricRecorder
from ..ml_type import MachineLearningPhase
from .amp import AMP
from .cudnn import CUDNNHook
from .debugger import Debugger
from .profiler import Profiler


class HookConfig:
    def __init__(self) -> None:
        self.debug = False
        self.profile = False
        self.use_amp = torch.cuda.is_available()
        self.save_performance_metric = False
        self.benchmark_cudnn: bool = True

    def append_hooks(self, executor) -> None:
        if executor.phase == MachineLearningPhase.Training:
            if torch.cuda.is_available():
                executor.enable_or_disable_hook("AMP", self.use_amp, AMP())
        executor.enable_or_disable_hook("debugger", self.debug, Debugger())
        executor.enable_or_disable_hook("profiler", self.profile, Profiler())
        if torch.cuda.is_available():
            executor.enable_or_disable_hook("cudnn", self.benchmark_cudnn, CUDNNHook())
        executor.enable_or_disable_hook(
            "performance_metric_recorder",
            self.save_performance_metric,
            PerformanceMetricRecorder(),
        )
