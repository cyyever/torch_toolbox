#!/usr/bin/env python3


import torch

if torch.cuda.is_available():
    import pynvml

# import os

from cyy_naive_lib.log import get_logger


def get_cuda_device_memory_info(consider_cache: bool = False) -> dict:
    result = {}
    pynvml.nvmlInit()
    for device_idx in range(torch.cuda.device_count()):
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


def get_cpu_device() -> torch.device:
    return torch.device("cpu")


def get_cuda_devices() -> list[torch.device]:
    device_count = torch.cuda.device_count()
    assert device_count > 0
    return [
        torch.device(f"cuda:{device_id}") for device_id in get_cuda_device_memory_info()
    ]


def get_devices() -> list[torch.device]:
    if torch.cuda.is_available():
        return get_cuda_devices()
    if torch.backends.mps.is_available():
        return [torch.device("mps")]
    return [torch.device("cpu")]


class CUDADeviceRoundRobinAllocator:
    def __init__(self):
        self.__devices = get_cuda_devices()
        self.__idx = 0

    def get_device(self):
        device = self.__devices[self.__idx]
        self.__idx += 1
        if self.__idx >= len(self.__devices):
            self.__idx = 0
        return device


class CUDADeviceGreedyAllocator:
    def get_devices(self, max_needed_bytes):
        memory_info = get_cuda_device_memory_info(consider_cache=True)
        memory_to_device = {}
        for device_id, info in memory_info.items():
            if (
                max_needed_bytes is not None
                and memory_info[device_id].free < max_needed_bytes
            ):
                continue
            if info.free not in memory_to_device:
                memory_to_device[info.free] = []
            memory_to_device[info.free].append(torch.device(f"cuda:{device_id}"))
        devices = []
        for k in reversed(sorted(memory_to_device.keys())):
            devices += memory_to_device[k]
        return devices

    def get_device(self, max_needed_bytes=None):
        devices = self.get_devices(max_needed_bytes=max_needed_bytes)
        if not devices:
            return None
        return devices[0]


def get_device(max_needed_bytes=None, use_cuda_only: bool = False) -> torch.device:
    if torch.cuda.is_available():
        device = CUDADeviceGreedyAllocator().get_device(
            max_needed_bytes=max_needed_bytes
        )
        if device is not None:
            return device
        if use_cuda_only:
            raise RuntimeError("no cuda device avaiable")
        get_logger().warning(
            "cuda device is unavailable, max_needed_bytes is %s, switch to CPU",
            max_needed_bytes,
        )
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return get_cpu_device()
