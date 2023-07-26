from typing import Any

import torch
from cyy_naive_lib.log import get_logger

from .dependency import has_pynvml

if has_pynvml:
    import pynvml


def get_cuda_memory_info(
    device_idx: int | None = None, consider_cache: bool = False
) -> dict:
    assert torch.cuda.is_available()
    assert has_pynvml
    result = {}
    pynvml.nvmlInit()
    if device_idx is not None:
        device_indices = [device_idx]
    else:
        device_indices = list(range(torch.cuda.device_count()))
    for device_idx in device_indices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        mode = pynvml.nvmlDeviceGetComputeMode(handle)
        if mode == pynvml.NVML_COMPUTEMODE_EXCLUSIVE_PROCESS:
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if processes:
                continue
                # if processes[0].pid != os.getpid():
                #     continue
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if consider_cache:
            cache_size = torch.cuda.memory_reserved(device=device_idx)
            info.used -= cache_size
            info.free += cache_size
        result[device_idx] = info
    pynvml.nvmlShutdown()
    return result


def get_device_memory_info(device: torch.device, consider_cache: bool = False) -> dict:
    match device.type.lower():
        case "cuda":
            return get_cuda_memory_info(
                device_idx=device.index, consider_cache=consider_cache
            )
    raise NotImplementedError()


def get_cpu_device() -> torch.device:
    return torch.device("cpu")


def __get_cuda_devices() -> list[torch.device]:
    device_count = torch.cuda.device_count()
    assert device_count > 0
    return [torch.device(f"cuda:{device_id}") for device_id in get_cuda_memory_info()]


def get_devices(
    max_needed_bytes: None | int = None, disable_cpu: bool = False
) -> list[torch.device]:
    if torch.cuda.is_available():
        devices = CUDADeviceGreedyAllocator.get_devices(
            max_needed_bytes=max_needed_bytes
        )
        if devices:
            return devices
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")
    if disable_cpu:
        raise RuntimeError("only CPU device is available")
    get_logger().warning("max_needed_bytes is %s, switch to CPU", max_needed_bytes)
    return [get_cpu_device()]


def get_device(**kwargs: Any) -> torch.device:
    return get_devices(**kwargs)[0]


class CUDADeviceRoundRobinAllocator:
    def __init__(self) -> None:
        self.__devices = __get_cuda_devices()
        self.__idx = 0

    def get_device(self) -> torch.device:
        device = self.__devices[self.__idx]
        self.__idx += 1
        if self.__idx >= len(self.__devices):
            self.__idx = 0
        return device


class CUDADeviceGreedyAllocator:
    @classmethod
    def get_devices(cls, max_needed_bytes: int | None) -> list[torch.device]:
        memory_info = get_cuda_memory_info(consider_cache=True)
        memory_to_device: dict = {}
        for device_id, info in memory_info.items():
            if max_needed_bytes is not None and info.free < max_needed_bytes:
                continue
            if info.free not in memory_to_device:
                memory_to_device[info.free] = []
            memory_to_device[info.free].append(torch.device(f"cuda:{device_id}"))
        devices = []
        for k in sorted(memory_to_device.keys(), reverse=True):
            devices += memory_to_device[k]
        return devices

    @classmethod
    def get_device(cls, max_needed_bytes: int | None = None) -> None | torch.device:
        devices = cls.get_devices(max_needed_bytes=max_needed_bytes)
        if not devices:
            return None
        return devices[0]
