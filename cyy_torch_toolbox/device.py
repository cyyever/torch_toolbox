#!/usr/bin/env python3

import pynvml
import torch


def get_cpu_device():
    return torch.device("cpu")


def get_cuda_devices():
    device_count = torch.cuda.device_count()
    assert device_count > 0
    devices = []
    for device_id in range(device_count):
        devices.append(torch.device("cuda:" + str(device_id)))
    return devices


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


class CudaDeviceSmartAllocator:
    def __init__(self, max_needed_bytes):
        self.__devices = get_cuda_devices()
        self.__cnts: dict = dict()
        for device in self.__devices:
            self.__cnts[device] = 0
        self.__max_need_bytes = max_needed_bytes

    def get_device(self):
        pynvml.nvmlInit()
        for device in sorted(self.__cnts.keys(), key=lambda x: self.__cnts[x]):
            h = pynvml.nvmlDeviceGetHandleByIndex(device.index)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            if info.free >= self.__max_need_bytes:
                self.__cnts[device] += 1
                pynvml.nvmlShutdown()
                return device
        pynvml.nvmlShutdown()
        return None
