import functools
import pickle
from typing import Any

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


def recursive_tensor_op(data, fun, **kwargs) -> Any:
    match data:
        case torch.Tensor():
            return fun(data, **kwargs)
        case list():
            for idx, element in enumerate(data):
                data[idx] = recursive_tensor_op(element, fun, **kwargs)
            return data
        case tuple():
            return tuple(recursive_tensor_op(list(data), fun, **kwargs))
        case dict():
            for k, v in data.items():
                data[k] = recursive_tensor_op(v, fun, **kwargs)
            return data
        case functools.partial():
            return functools.partial(
                data.func,
                *recursive_tensor_op(data.args, fun, **kwargs),
                **recursive_tensor_op(data.keywords, fun, **kwargs)
            )
    if has_hugging_face:
        match data:
            case transformers.tokenization_utils_base.BatchEncoding():
                data.data = recursive_tensor_op(data.data, fun, **kwargs)
                return data
    return data


def tensor_to(data, non_blocking=False, check_pin=False, **kwargs):
    def fun(data, check_pin, **kwargs):
        if check_pin and str(data.device) == "cpu" and not data.is_pinned():
            raise RuntimeError("tensor is not pinned")
        return data.to(**kwargs)

    return recursive_tensor_op(
        data, fun, non_blocking=non_blocking, check_pin=check_pin, **kwargs
    )


def tensor_clone(data, detach=True):
    def fun(data, detach):
        new_data = data.clone()
        if detach:
            new_data = new_data.detach()
        return new_data

    return recursive_tensor_op(data, fun, detach=detach)
