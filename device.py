#!/usr/bin/env python3

import torch


def get_cpu_device():
    return torch.device("cpu")

def get_cuda_device(device_id=None):
    if device_id:
        return torch.device("cuda:"+str(device_id))
    return torch.device("cuda")

def get_device():
    if torch.cuda.is_available():
        return get_cuda_device()
    return torch.device("cpu")
