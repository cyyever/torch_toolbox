import functools
from typing import Any

import torch
from cyy_naive_lib.log import log_info

from . import Hook


class CUDAMemoryProfiler(Hook):
    def __init__(self, *args, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.__hooks: list[Any] = []
        self.__used_memory: list[tuple[str, float]] = []

    def _before_execute(self, **kwargs: Any) -> None:
        self.__hooks = []
        self.__used_memory = []

    def _before_batch(self, executor, batch_index, **kwargs: Any) -> None:
        if batch_index != 0:
            return
        assert not self.__hooks
        for module_name, module in executor.model.named_modules():
            if not module_name:
                continue
            if not any(True for _ in module.parameters()):
                continue
            cur_used_memory = torch.cuda.memory_allocated()
            self.__used_memory.append(("", float(cur_used_memory) / 1024 / 1024))
            self.__hooks.append(
                module.register_forward_hook(
                    functools.partial(
                        self.__compute_gpu_memory_assumption,
                        module_name,
                        len(self.__hooks),
                    )
                )
            )

    def __compute_gpu_memory_assumption(
        self,
        module_name: str,
        hook_idx: int,
        module: torch.nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        cur_used_memory = torch.cuda.memory_allocated()
        self.__used_memory.append((module_name, float(cur_used_memory) / 1024 / 1024))
        log_info(
            "%.1f MB CUDA memory is used for module %s",
            self.__used_memory[-1][1] - self.__used_memory[-2][1],
            self.__used_memory[-1][0],
        )
        self.__hooks[hook_idx].remove()
