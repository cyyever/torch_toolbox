import pynvml
import torch

from .base import MemoryInfo


def get_cuda_memory_info(
    device_idx: int | None = None, consider_cache: bool = True
) -> dict[torch.device, MemoryInfo]:
    assert torch.cuda.is_available()

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
            # PyTorch bug
            if cache_size <= info.used:
                # pylint: disable=no-member
                info.used -= cache_size
                # pylint: disable=no-member
                info.free += cache_size
        result[torch.device(f"cuda:{d_idx}")] = MemoryInfo(
            # pylint: disable=no-member
            used=info.used,
            # pylint: disable=no-member
            free=info.free,
            # pylint: disable=no-member
            total=info.total,
        )
    pynvml.nvmlShutdown()
    return result
