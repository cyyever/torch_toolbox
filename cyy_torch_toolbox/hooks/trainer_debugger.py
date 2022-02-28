import torch
from cyy_torch_toolbox.hook import Hook

from .gradient_sanitizer import GradientSanitizer


class TrainerDebugger(Hook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gradient_sanitizer = GradientSanitizer()

    def _before_execute(self, **kwargs):
        torch.autograd.set_detect_anomaly(True)
        torch.set_anomaly_enabled(True)
        torch.cuda.set_sync_debug_mode(1)

    def _after_execute(self, **kwargs):
        torch.autograd.set_detect_anomaly(False)
        torch.set_anomaly_enabled(False)
        torch.cuda.set_sync_debug_mode(0)
