#!/usr/bin/env python3

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


def get_cuda_device(device_id=None):
    if device_id:
        return torch.device("cuda:" + str(device_id))
    return torch.device("cuda")


def get_device():
    if torch.cuda.is_available():
        return get_cuda_device()
    return torch.device("cpu")
