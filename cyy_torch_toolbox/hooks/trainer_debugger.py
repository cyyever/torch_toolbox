import torch
from hook import Hook

from .cuda_memory_tracker import CUDAMemoryTracker
from .gradient_sanitizer import GradientSanitizer


class TrainerDebugger(Hook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gradient_sanitizer = GradientSanitizer()
        self.cuda_memory_tracker = CUDAMemoryTracker()

    def _before_execute(self, **kwargs):
        torch.set_anomaly_enabled(True)

    def _after_execute(self, **kwargs):
        torch.set_anomaly_enabled(False)
