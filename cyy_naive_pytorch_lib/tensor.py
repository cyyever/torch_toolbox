import pickle

import numpy as np
import torch
import torch.nn as nn


def cat_tensors_to_vector(tensors) -> torch.Tensor:
    return nn.utils.parameters_to_vector([t.reshape(-1) for t in tensors])


def load_tensor_dict(data: dict, tensor: torch.Tensor):
    bias = 0
    for name in sorted(data.keys()):
        shape = data[name].shape
        param_element_num = np.prod(shape)
        data[name] = tensor.narrow(0, bias, param_element_num).view(*shape)
        bias += param_element_num
    assert bias == tensor.shape[0]
    return data


def split_tensor_to_dict(name_and_shapes: list, tensor: torch.Tensor):
    data = dict()
    bias = 0
    for (name, shape) in name_and_shapes:
        param_element_num = np.prod(shape)
        data[name] = tensor.narrow(0, bias, param_element_num).view(*shape)
        bias += param_element_num
    assert bias == tensor.shape[0]
    return data


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device=device)
    if isinstance(data, list):
        for idx, d in enumerate(data):
            data[idx] = to_device(d, device)
        return data
    if isinstance(data, tuple):
        return tuple(to_device(list(data), device))
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = to_device(v, device)
        return data
    return data


def get_tensor_serialization_size(data):
    return len(pickle.dumps(data))
