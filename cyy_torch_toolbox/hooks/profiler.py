from hook import Hook

from .cuda_memory_profiler import CUDAMemoryProfiler
from .dataloader_profiler import DataLoaderProfiler


class Profiler(Hook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cuda_memory_profiler = CUDAMemoryProfiler()
        self.dataloader_profiler = DataLoaderProfiler()
