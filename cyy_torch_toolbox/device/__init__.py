from typing import Any, Self

import psutil
import torch

from .base import MemoryInfo

if torch.cuda.is_available():
    from .cuda import get_cuda_memory_info
else:

    def get_cuda_memory_info(**kwargs: Any) -> dict:
        raise NotImplementedError()


def get_device_memory_info(
    device: torch.device | None = None, consider_cache: bool = False
) -> dict[torch.device, MemoryInfo]:
    device_type: str = ""
    device_idx: int | None = None
    if device is not None:
        device_type = device.type.lower()
        device_idx = device.index
    else:
        if torch.cuda.is_available():
            device_type = "cuda"
        elif torch.backends.mps.is_available():
            device_type = "mps"
        elif torch.xpu.is_available():
            device_type = "xpu"
        else:
            device_type = "cpu"
    match device_type:
        case "cuda":
            return get_cuda_memory_info(
                device_idx=device_idx, consider_cache=consider_cache
            )
        case "cpu" | "mps" | "vulkan" | "xpu":
            device = torch.device(type=device_type, index=0)
            vm = psutil.virtual_memory()
            return {
                device: MemoryInfo(
                    free=vm.available,
                    total=vm.total,
                    used=vm.used,
                )
            }
    raise NotImplementedError(device_type)


def set_device(device: torch.device) -> None:
    match device.type.lower():
        case "cuda":
            torch.cuda.set_device(device)
        case "xpu":
            torch.xpu.set_device(device)


class DeviceGreedyAllocator:
    @classmethod
    def get_devices(cls, max_needed_bytes: int | None = None) -> list[torch.device]:
        memory_info = get_device_memory_info()
        memory_to_device: dict = {}
        for device, info in memory_info.items():
            if max_needed_bytes is not None and info.free < max_needed_bytes:
                continue
            if info.used / info.total > 0.95:
                continue
            if info.free not in memory_to_device:
                memory_to_device[info.free] = []
            memory_to_device[info.free].append(device)
        devices: list[torch.device] = []
        for k in sorted(memory_to_device.keys(), reverse=True):
            devices += memory_to_device[k]
        return devices

    @classmethod
    def get_device(cls, **kwargs: Any) -> torch.device:
        return cls.get_devices(**kwargs)[0]


class SyncedStreamContext:
    def __init__(self, stream: torch.cpu.Stream | torch.Stream) -> None:
        self.__stream = stream

    def __enter__(self) -> Self:
        if isinstance(self.__stream, torch.Stream):
            device = self.__stream.device
            if "cuda" in device.type.lower():
                self.__stream.wait_stream(torch.cuda.default_stream(device))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if hasattr(self.__stream, "synchronize"):
            self.__stream.synchronize()
        if hasattr(self.__stream, "query"):
            assert self.__stream.query()


class DefaultDeviceContext:
    def __init__(self, device: torch.device) -> None:
        self.__device = device
        self.__previous_device: torch.device | None = None

    def __enter__(self) -> Self:
        self.__previous_device = torch.get_default_device()
        torch.set_default_device(self.__device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        torch.set_default_device(self.__previous_device)
