from hook import Hook

from .cuda_memory_profiler import CUDAMemoryProfiler
from .dataloader_profiler import DataloaderProfiler


class Profiler(Hook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__cuda_memory_profiler = CUDAMemoryProfiler()
        self.__dataloader_profiler = DataloaderProfiler()
