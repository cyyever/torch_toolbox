from typing import Any

import torch

from . import Hook
from .cuda_stream_profiler import CUDAStreamProfiler

# from .cuda_memory_profiler import CUDAMemoryProfiler


class Profiler(Hook):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if torch.cuda.is_available():
            self.cuda_stream_profiler = CUDAStreamProfiler()
            # self.cuda_memory_profiler = CUDAMemoryProfiler()
