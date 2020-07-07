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


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
