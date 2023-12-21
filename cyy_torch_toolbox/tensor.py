import dataclasses
import functools
import pickle
from collections.abc import Iterable
from typing import Any, Callable

import numpy
import torch
from cyy_naive_lib.algorithm.mapping_op import (
    get_mapping_items_by_key_order, get_mapping_values_by_key_order)


def cat_tensors_to_vector(tensors: Iterable) -> torch.Tensor:
    return torch.cat([t.view(-1) for t in tensors])


def cat_tensor_dict(tensor_dict: dict) -> torch.Tensor:
    return cat_tensors_to_vector(get_mapping_values_by_key_order(tensor_dict))


def decompose_like_tensor_dict(tensor_dict: dict, tensor: torch.Tensor) -> dict:
    result = {}
    bias = 0
    for key, component in get_mapping_items_by_key_order(tensor_dict):
        param_element_num = numpy.prod(component.shape)
        result[key] = tensor[bias: bias + param_element_num].view(*component.shape)
        bias += param_element_num
    assert bias == tensor.shape[0]
    return result


def decompose_tensor_to_list(shapes: list, tensor: torch.Tensor) -> list:
    result = []
    bias = 0
    for shape in shapes:
        param_element_num = numpy.prod(shape)
        result.append(tensor[bias: bias + param_element_num].view(*shape))
        bias += param_element_num
    assert bias == tensor.shape[0]
    return result


def get_tensor_serialization_size(data):
    return len(pickle.dumps(data))


class __RecursiveCheckPoint:
    def __init__(self, data: Any) -> None:
        self.data: Any = data


def recursive_tensor_op(data: Any, fun: Callable, **kwargs: Any) -> Any:
    match data:
        case __RecursiveCheckPoint():
            if kwargs.pop("__check_recursive_point", False):
                return fun(data.data, **kwargs)
            return data
        case torch.Tensor():
            if kwargs.get("__check_recursive_point", False):
                return data
            return fun(data, **kwargs)
        case list():
            return [recursive_tensor_op(element, fun, **kwargs) for element in data]
        case tuple():
            return tuple(
                recursive_tensor_op(element, fun, **kwargs) for element in data
            )
        case dict():
            return {k: recursive_tensor_op(v, fun, **kwargs) for k, v in data.items()}
        case functools.partial():
            return functools.partial(
                data.func,
                *recursive_tensor_op(data.args, fun, **kwargs),
                **recursive_tensor_op(data.keywords, fun, **kwargs),
            )
    dataclass_fileds = None
    try:
        dataclass_fileds = dataclasses.fields(data)
    except BaseException:
        dataclass_fileds = None

    if dataclass_fileds is not None:
        for field in dataclasses.fields(data):
            setattr(
                data,
                field.name,
                recursive_tensor_op(getattr(data, field.name), fun, **kwargs),
            )
    elif hasattr(data, "data"):
        data.data = recursive_tensor_op(data.data, fun, **kwargs)
    return data


def tensor_to(
    data: Any, non_blocking: bool = True, check_slowdown: bool = False, **kwargs: Any
) -> Any:
    def fun(data, check_slowdown, **kwargs):
        if check_slowdown:
            device = kwargs.get("device", None)
            non_blocking = kwargs.get("non_blocking", True)
            if (
                str(data.device) == "cpu"
                and device is not None
                and str(device) != str(data.device)
            ):
                # if not data.is_pinned():
                #     raise RuntimeError("tensor is not pinned")
                if not non_blocking:
                    raise RuntimeError(
                        "copy is blocking",
                    )
            else:
                if device is not None and not kwargs.get("non_blocking", True):
                    raise RuntimeError(
                        "device to device copy is blocking",
                    )
            assert str(device) != str(data.device)
        return data.to(**kwargs)

    return recursive_tensor_op(
        data, fun, non_blocking=non_blocking, check_slowdown=check_slowdown, **kwargs
    )


def tensor_clone(data: Any, detach: bool = True) -> Any:
    def fun(data, detach):
        new_data = data.clone()
        if detach:
            new_data = new_data.detach()
        return new_data

    return recursive_tensor_op(data, fun, detach=detach)


def assemble_tensors(data: Any) -> tuple[torch.Tensor | None, Any]:
    tensor_list = []
    offset = 0

    def fun(data: torch.Tensor) -> __RecursiveCheckPoint:
        nonlocal offset
        if data.numel() == 0:
            return __RecursiveCheckPoint(data=(data,))
        shape = list(data.shape)
        if not shape:
            return __RecursiveCheckPoint(data=(data.item(),))
        if data.dtype != torch.float32:
            return __RecursiveCheckPoint(data=(data,))
        old_offset = offset
        tensor_list.append(data.view(-1))
        offset += data.numel()
        return __RecursiveCheckPoint(data=(shape, old_offset))

    res = recursive_tensor_op(data, fun)
    if offset == 0:
        assert not tensor_list
        return None, res
    assert tensor_list
    return cat_tensors_to_vector(tensor_list), res


def disassemble_tensor(
    concatenated_tensor: torch.Tensor, data: Any, clone: bool = True
) -> Any:
    def fun(data: __RecursiveCheckPoint) -> Any:
        if len(data) == 1:
            return data[0]
        shape, offset = data
        tensor = concatenated_tensor[offset: offset + numpy.prod(shape)].view(*shape)
        if clone:
            tensor = tensor.clone()
        return tensor

    if concatenated_tensor is None:
        return data

    return recursive_tensor_op(data, fun, __check_recursive_point=True)
