#!/usr/bin/env python3


import torch

if torch.cuda.is_available():
    import pynvml

from cyy_naive_lib.log import get_logger


def get_cpu_device():
    return torch.device("cpu")


def get_cuda_devices():
    device_count = torch.cuda.device_count()
    assert device_count > 0
    devices = []
    for device_id in range(device_count):
        devices.append(torch.device("cuda:" + str(device_id)))
    return devices


def get_devices():
    if torch.cuda.is_available():
        return get_cuda_devices()
    return [torch.device("cpu")]


def put_data_to_device(data, device, non_blocking=False):
    match data:
        case torch.Tensor():
            return data.to(device, non_blocking=non_blocking)
        case list():
            for idx, element in enumerate(data):
                data[idx] = put_data_to_device(
                    element, device, non_blocking=non_blocking
                )
            return data
        case tuple():
            return tuple(
                put_data_to_device(list(data), device, non_blocking=non_blocking)
            )
        case dict():
            for k, v in data.items():
                data[k] = put_data_to_device(v, device, non_blocking=non_blocking)
            return data
    return data


class CudaDeviceRoundRobinAllocator:
    def __init__(self):
        self.__devices = get_cuda_devices()
        self.__idx = 0

    def get_device(self):
        device = self.__devices[self.__idx]
        self.__idx += 1
        if self.__idx >= len(self.__devices):
            self.__idx = 0
        return device


class CudaDeviceGreedyAllocator:
    def __init__(self):
        self.__devices = get_cuda_devices()
        self.__free_memory_dict: dict = {}
        for device in self.__devices:
            self.__free_memory_dict[device] = 0

    def __refresh_memory_info(self):
        torch.cuda.empty_cache()
        pynvml.nvmlInit()
        for device in self.__free_memory_dict:
            h = pynvml.nvmlDeviceGetHandleByIndex(device.index)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            self.__free_memory_dict[device] = info.free + torch.cuda.memory_reserved(
                device=device
            )
        pynvml.nvmlShutdown()

    def get_devices(self, max_needed_bytes):
        return [
            device
            for device, memory in self.__sort_devices()
            if max_needed_bytes is None or memory >= max_needed_bytes
        ]

    def get_device(self, max_needed_bytes=None):
        devices = self.get_devices(max_needed_bytes=max_needed_bytes)
        if not devices:
            return None
        # cuda_device = torch.cuda.current_device()
        # if cuda_device >= 0:
        #     if cuda_device in {d.index for d in devices}:
        #         return torch.device(f"cuda:{cuda_device}")
        # get_logger().debug(
        #     "current_device %s choose device %s", cuda_device, devices[0]
        # )
        return devices[0]

    def __sort_devices(self) -> list:
        self.__refresh_memory_info()
        return sorted(
            self.__free_memory_dict.items(), key=lambda item: item[1], reverse=True
        )


def get_device(max_needed_bytes=None, use_cuda_only=False):
    if torch.cuda.is_available():
        device = CudaDeviceGreedyAllocator().get_device(
            max_needed_bytes=max_needed_bytes
        )
        if device is not None:
            return device
        if use_cuda_only:
            raise RuntimeError("no cuda device avaiable")
        get_logger().warning(
            "cuda device is unavaiable, max_needed_bytes is %s, switch to CPU",
            max_needed_bytes,
        )
    return get_cpu_device()
