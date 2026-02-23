from typing import Any

import torch

from . import Hook


class AcceleratorStreamProfiler(Hook):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.__device_module: Any = None

    def _before_execute(self, executor, **kwargs: Any) -> None:
        mod = torch.get_device_module(executor.device)
        if hasattr(mod, "set_sync_debug_mode"):
            mod.set_sync_debug_mode("warn")
            self.__device_module = mod

    def _after_execute(self, **kwargs: Any) -> None:
        if self.__device_module is not None:
            self.__device_module.set_sync_debug_mode("default")
            self.__device_module = None
