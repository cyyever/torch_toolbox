from dataclasses import dataclass
from typing import Any

import psutil
import torch
from cyy_naive_lib.log import get_logger

from .dependency import has_pynvml

if has_pynvml:
    import pynvml


@dataclass(kw_only=True)
class MemoryInfo:
    total: int
    free: int
    used: int


def _get_cuda_memory_info(
    device_idx: int | None = None, consider_cache: bool = True
) -> dict[torch.device, MemoryInfo]:
    assert torch.cuda.is_available()
    assert has_pynvml
    result = {}
    pynvml.nvmlInit()
    if device_idx is not None:
        device_indices = [device_idx]
    else:
        device_indices = list(range(torch.cuda.device_count()))
    for d_idx in device_indices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(d_idx)
        mode = pynvml.nvmlDeviceGetComputeMode(handle)
        if mode == pynvml.NVML_COMPUTEMODE_EXCLUSIVE_PROCESS:
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if processes:
                continue
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if consider_cache:
            cache_size = torch.cuda.memory_reserved(device=d_idx)
            info.used -= cache_size
            info.free += cache_size
        result[torch.device(f"cuda:{d_idx}")] = MemoryInfo(
            used=info.used,
            free=info.free,
            total=info.total,
        )
    pynvml.nvmlShutdown()
    return result


def get_device_memory_info(
    device: torch.device | None = None, consider_cache: bool = True
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
        # elif torch.is_vulkan_available():
        #     device_type = "vulkan"
        else:
            device_type = "cpu"
    match device_type:
        case "cuda":
            return _get_cuda_memory_info(
                device_idx=device_idx, consider_cache=consider_cache
            )
        case "cpu" | "mps" | "vulkan":
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


def get_cpu_device() -> torch.device:
    return torch.device("cpu")


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
        devices = []
        for k in sorted(memory_to_device.keys(), reverse=True):
            devices += memory_to_device[k]
        return devices

    @classmethod
    def get_device(cls, **kwargs: Any) -> torch.device:
        return cls.get_devices(**kwargs)[0]


def get_devices(max_needed_bytes: None | int = None) -> list[torch.device]:
    devices = DeviceGreedyAllocator.get_devices(max_needed_bytes=max_needed_bytes)
    if "cpu" not in devices[0].type.lower():
        return devices
    get_logger().error(
        "max_needed_bytes is %s, only CPU device is available, which we don't want",
        max_needed_bytes,
    )
    return devices


def get_device(**kwargs: Any) -> torch.device:
    return get_devices(**kwargs)[0]
