from typing import Any

import torch.autograd
from cyy_naive_lib.log import log_warning

from . import Hook
from .gradient_sanitizer import GradientSanitizer


class Debugger(Hook):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gradient_sanitizer = GradientSanitizer()

    def _before_execute(self, executor, **kwargs: Any) -> None:
        torch.autograd.set_detect_anomaly(True)
        log_warning("model executor in debugging mode")

    def _after_execute(self, **kwargs: Any) -> None:
        torch.autograd.set_detect_anomaly(False)
