import torch
import torch.cuda

from cyy_torch_toolbox.hooks.amp import AMP
from cyy_torch_toolbox.hooks.cudnn import CUDNNHook
from cyy_torch_toolbox.hooks.debugger import Debugger
from cyy_torch_toolbox.hooks.profiler import Profiler
from cyy_torch_toolbox.metric_visualizers.performance_metric_recorder import \
    PerformanceMetricRecorder
from cyy_torch_toolbox.ml_type import MachineLearningPhase


class HookConfig:
    def __init__(self):
        self.debug = False
        self.profile = False
        self.use_amp = torch.cuda.is_available()
        self.save_performance_metric = False
        self.benchmark_cudnn: bool = True

    def append_hooks(self, executor) -> None:
        if executor.phase == MachineLearningPhase.Training:
            if torch.cuda.torch.cuda.is_available():
                executor.enable_or_disable_hook("AMP", self.use_amp, AMP())
        executor.enable_or_disable_hook("debugger", self.debug, Debugger())
        executor.enable_or_disable_hook("profiler", self.profile, Profiler())
        if torch.cuda.torch.cuda.is_available():
            executor.enable_or_disable_hook("cudnn", self.benchmark_cudnn, CUDNNHook())
        executor.enable_or_disable_hook(
            "performance_metric_recorder",
            self.save_performance_metric,
            PerformanceMetricRecorder(),
        )
