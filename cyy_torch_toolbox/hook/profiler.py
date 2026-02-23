from typing import Any

import torch

from . import Hook
from .accelerator_stream_profiler import AcceleratorStreamProfiler


class Profiler(Hook):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if torch.accelerator.is_available():
            self.accelerator_stream_profiler = AcceleratorStreamProfiler()
