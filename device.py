#!/usr/bin/env python3

import torch


def get_cpu_device():
    return torch.device("cuda")


def get_device():
    if torch.cuda.is_available():
        return get_cpu_device()
    return torch.device("cpu")
