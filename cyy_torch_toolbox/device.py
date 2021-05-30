#!/usr/bin/env python3

import time

import pynvml
import torch
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


def put_data_to_device(data, device=None):
    if device is None:
        device = get_device()
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, list):
        for idx, element in enumerate(data):
            data[idx] = put_data_to_device(element, device)
        return data
    if isinstance(data, tuple):
        return tuple(put_data_to_device(list(data), device))
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = put_data_to_device(v, device)
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
        self.__free_memory_dict: dict = dict()
        for device in self.__devices:
            self.__free_memory_dict[device] = 0
        self.__last_query_time = None

    def __refresh_memory_info(self):
        if (
            self.__last_query_time is not None
            and time.time() < self.__last_query_time + 60 * 10
        ):
            return
        pynvml.nvmlInit()
        for device in self.__free_memory_dict:
            h = pynvml.nvmlDeviceGetHandleByIndex(device.index)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            self.__free_memory_dict[device] = info.free
        pynvml.nvmlShutdown()
        self.__last_query_time = time.time()

    def get_device(self, max_needed_bytes=None):
        for device in self.__sort_devices():
            if (
                max_needed_bytes is not None
                and self.__free_memory_dict[device] < max_needed_bytes
            ):
                continue
            return device
        return None

    def __sort_devices(self) -> list:
        self.__refresh_memory_info()
        return sorted(
            self.__free_memory_dict.keys(),
            key=lambda x: self.__free_memory_dict[x],
            reverse=True,
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
