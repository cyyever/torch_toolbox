import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.hook import Hook

from .gradient_sanitizer import GradientSanitizer


class Debugger(Hook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gradient_sanitizer = GradientSanitizer()

    def _before_execute(self, **kwargs):
        torch.autograd.set_detect_anomaly(True)
        torch.set_anomaly_enabled(True)
        if torch.cuda.is_available():
            torch.cuda.set_sync_debug_mode(1)
        get_logger().warning("model executor in in debugging mode")

    def _after_execute(self, **kwargs):
        torch.autograd.set_detect_anomaly(False)
        torch.set_anomaly_enabled(False)
        if torch.cuda.is_available():
            torch.cuda.set_sync_debug_mode(0)
