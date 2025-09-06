import os

import pynvml
import torch

from .base import MemoryInfo


def get_cuda_memory_info(
    device_idx: int | None = None, consider_cache: bool = True
) -> dict[torch.device, MemoryInfo]:
    pynvml.nvmlInit()
    device_cnt = pynvml.nvmlDeviceGetCount()
    assert device_cnt > 0

    device_map = {i: i for i in range(device_cnt)}
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
    device_list = list(range(device_cnt))
    if cuda_visible_devices is not None:
        device_list = sorted([int(d) for d in cuda_visible_devices.split(",")])
        device_map = {
            device_id: v_device_id for v_device_id, device_id in enumerate(device_list)
        }

    result = {}
    for d_idx in device_list:
        v_d_idx = device_map[d_idx]
        if device_idx is not None and v_d_idx != device_idx:
            continue
        handle = pynvml.nvmlDeviceGetHandleByIndex(d_idx)
        mode = pynvml.nvmlDeviceGetComputeMode(handle)
        if mode == pynvml.NVML_COMPUTEMODE_EXCLUSIVE_PROCESS:
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if processes:
                continue
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = info.used  # noqa
        free = info.free  # noqa
        total = info.total  # noqa
        assert isinstance(used, int)
        assert isinstance(free, int)
        assert isinstance(total, int)
        if consider_cache:
            cache_size = torch.cuda.memory_reserved(device=v_d_idx)
            # PyTorch bug
            if cache_size <= used:
                used -= cache_size
                free += cache_size
        result[torch.device(f"cuda:{v_d_idx}")] = MemoryInfo(
            used=used,
            free=free,
            total=total,
        )
    pynvml.nvmlShutdown()
    return result
