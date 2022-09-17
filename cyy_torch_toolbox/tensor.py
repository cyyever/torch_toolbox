import pickle

import numpy
import torch
import torch.nn as nn

try:
    import transformers

    has_hugging_face = True
except ModuleNotFoundError:
    has_hugging_face = False


def cat_tensors_to_vector(tensors: list) -> torch.Tensor:
    return nn.utils.parameters_to_vector([t.reshape(-1) for t in tensors])


def load_tensor_dict(shapes: dict, tensor: torch.Tensor) -> dict:
    bias = 0
    result = {}
    for name in sorted(shapes.keys()):
        shape = shapes[name]
        param_element_num = numpy.prod(shape)
        result[name] = tensor.narrow(0, bias, param_element_num).view(*shape)
        bias += param_element_num
    assert bias == tensor.shape[0]
    return result


def split_tensor_to_dict(name_and_shapes: list, tensor: torch.Tensor) -> dict:
    data = {}
    bias = 0
    for (name, shape) in name_and_shapes:
        param_element_num = numpy.prod(shape)
        data[name] = tensor.narrow(0, bias, param_element_num).view(*shape)
        bias += param_element_num
    assert bias == tensor.shape[0]
    return data


def split_tensor_to_list(shapes: list, tensor: torch.Tensor) -> list:
    data = []
    bias = 0
    for shape in shapes:
        param_element_num = numpy.prod(shape)
        data.append(tensor.narrow(0, bias, param_element_num).view(*shape))
        bias += param_element_num
    assert bias == tensor.shape[0]
    return data


def get_tensor_serialization_size(data):
    return len(pickle.dumps(data))


def tensor_to(data, non_blocking=False, **kwargs):
    match data:
        case torch.Tensor():
            return data.to(non_blocking=non_blocking, **kwargs)
        case list():
            for idx, element in enumerate(data):
                data[idx] = tensor_to(element, non_blocking=non_blocking, **kwargs)
            return data
        case tuple():
            return tuple(tensor_to(list(data), non_blocking=non_blocking, **kwargs))
        case dict():
            for k, v in data.items():
                data[k] = tensor_to(v, non_blocking=non_blocking, **kwargs)
            return data
    if has_hugging_face:
        match data:
            case transformers.tokenization_utils_base.BatchEncoding():
                data.data = tensor_to(data.data, non_blocking=non_blocking, **kwargs)
                return data
    return data
