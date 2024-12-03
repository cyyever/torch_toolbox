import torch

from . import Hook
from .cuda_memory_profiler import CUDAMemoryProfiler
from .cuda_stream_profiler import CUDAStreamProfiler


class Profiler(Hook):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if torch.cuda.is_available():
            self.cuda_stream_profiler = CUDAStreamProfiler()
            self.cuda_memory_profiler = CUDAMemoryProfiler()
