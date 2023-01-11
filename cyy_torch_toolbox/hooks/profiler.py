import torch
from cyy_torch_toolbox.hook import Hook

from .cuda_stream_profiler import CUDAStreamProfiler
# from .cuda_memory_profiler import CUDAMemoryProfiler


class Profiler(Hook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if torch.cuda.is_available():
            self.__cuda_stream_profiler = CUDAStreamProfiler()
            # self.__cuda_memory_profiler = CUDAMemoryProfiler()
