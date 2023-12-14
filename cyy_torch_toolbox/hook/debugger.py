from typing import Any

import torch
from cyy_naive_lib.log import get_logger

from . import Hook
from .gradient_sanitizer import GradientSanitizer


class Debugger(Hook):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gradient_sanitizer = GradientSanitizer()

    def _before_execute(self, executor, **kwargs: Any) -> None:
        torch.autograd.set_detect_anomaly(True)
        get_logger().warning("model executor in debugging mode")

    def _after_execute(self, **kwargs: Any) -> None:
        torch.autograd.set_detect_anomaly(False)
