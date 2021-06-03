import torch
from hook import ComposeHook, Hook

from .gradient_sanitizer import GradientSanitizer

# from .memory_tracker import MemoryTracker


class TrainerDebugger(ComposeHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gradient_sanitizer = GradientSanitizer()

    def _before_execute(self, **kwargs):
        torch.set_anomaly_enabled(True)

    def _after_execute(self, **kwargs):
        torch.set_anomaly_enabled(False)
