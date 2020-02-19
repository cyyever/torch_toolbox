#!/usr/bin/env python3

import torch


def get_cpu_device():
    return torch.device("cpu")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
