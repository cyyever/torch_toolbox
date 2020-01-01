#!/usr/bin/env python3

import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
