import functools

import torch
from cyy_naive_lib.log import get_logger
from hook import Hook
from torch.nn.modules.activation import ReLU


class CUDAMemoryProfiler(Hook):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__hooks = []
        self.__used_memory = None

    def _before_batch(self, **kwargs):
        assert not self.__hooks
        self.__used_memory = None
        model_executor = kwargs["model_executor"]
        if kwargs["batch_index"] != 2:
            return
        for module_name, module in model_executor.model.named_modules():
            self.__hooks.append(
                module.register_forward_hook(
                    functools.partial(self.__compute_gpu_memory_assumption, module_name)
                )
            )

    def __compute_gpu_memory_assumption(self, module_name, module, _, __):
        if not module_name:
            return
        if isinstance(module, ReLU):
            return
        cur_used_memory = torch.cuda.memory_allocated()
        if self.__used_memory is None:
            get_logger().info(
                "%.1f MB CUDA memory is used after first module %s",
                float(cur_used_memory) / 1024 / 1024,
                module_name,
            )
        else:
            get_logger().info(
                "%.1f MB CUDA memory is used after module %s, difference is %.1f MB",
                float(cur_used_memory) / 1024 / 1024,
                module_name,
                float(cur_used_memory - self.__used_memory) / 1024 / 1024,
            )
        self.__used_memory = cur_used_memory

    def _after_batch(self, **kwargs):
        for h in self.__hooks:
            h.remove()
        self.__hooks = []
        self.__used_memory = None
