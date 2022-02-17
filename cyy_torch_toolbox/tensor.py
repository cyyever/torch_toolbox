import pickle

import torch
import torch.nn as nn


def cat_tensors_to_vector(tensors) -> torch.Tensor:
    return nn.utils.parameters_to_vector([t.reshape(-1) for t in tensors])


def load_tensor_dict(data: dict, tensor: torch.Tensor):
    bias = 0
    for name in sorted(data.keys()):
        shape = data[name].shape
        param_element_num = torch.prod(shape)
        data[name] = tensor.narrow(0, bias, param_element_num).view(*shape)
        bias += param_element_num
    assert bias == tensor.shape[0]
    return data


def split_tensor_to_dict(name_and_shapes: list, tensor: torch.Tensor) -> dict:
    data = {}
    bias = 0
    for (name, shape) in name_and_shapes:
        param_element_num = torch.prod(shape)
        data[name] = tensor.narrow(0, bias, param_element_num).view(*shape)
        bias += param_element_num
    assert bias == tensor.shape[0]
    return data


def split_tensor_to_list(shapes: list, tensor: torch.Tensor) -> torch.Tensor:
    data = []
    bias = 0
    for shape in shapes:
        param_element_num = torch.prod(shape)
        data.append(tensor.narrow(0, bias, param_element_num).view(*shape))
        bias += param_element_num
    assert bias == tensor.shape[0]
    return data


def get_tensor_serialization_size(data):
    return len(pickle.dumps(data))
