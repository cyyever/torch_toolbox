import functools
import pickle
from collections.abc import Iterable
from typing import Any, Callable

import numpy
import torch
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order

try:
    import transformers

    has_hugging_face = True
except ModuleNotFoundError:
    has_hugging_face = False


def cat_tensors_to_vector(tensors: Iterable) -> torch.Tensor:
    return torch.cat([t.view(-1) for t in tensors])


def cat_tensor_dict(tensor_dict: dict) -> torch.Tensor:
    return cat_tensors_to_vector(get_mapping_values_by_key_order(tensor_dict))


def load_tensor_dict_from_seq(shapes: dict, tensor_seq: list) -> dict:
    result = {}
    for name in sorted(shapes.keys()):
        shape = shapes[name]
        assert tensor_seq[0].shape == shape
        result[name] = tensor_seq[0]
        tensor_seq = tensor_seq[1:]
    return result


def load_tensor_dict(shapes: dict, tensor: torch.Tensor) -> dict:
    bias = 0
    result = {}
    total_size = tensor.numel()
    for name in sorted(shapes.keys()):
        shape = shapes[name]
        param_element_num = numpy.prod(shape)
        result[name] = tensor[bias: bias + param_element_num].view(*shape)
        bias += param_element_num
    assert bias == total_size
    return result


def decompose_tensor_to_dict(shapes: dict, tensor: torch.Tensor) -> dict:
    return load_tensor_dict(shapes, tensor)


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
    def __init__(self, data):
        self.data = data


def recursive_tensor_op(data: Any, fun: Callable, **kwargs: dict) -> Any:
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
            try:
                # we need to check in key order because that order may be useful
                keys = sorted(data.keys())
            except BaseException:
                keys = list(data.keys())
            return {k: recursive_tensor_op(data[k], fun, **kwargs) for k in keys}
        case functools.partial():
            return functools.partial(
                data.func,
                *recursive_tensor_op(data.args, fun, **kwargs),
                **recursive_tensor_op(data.keywords, fun, **kwargs)
            )
        # case _:
        #     print("unsupported tensor type", type(data))
    if has_hugging_face:
        match data:
            case transformers.tokenization_utils_base.BatchEncoding():
                data.data = recursive_tensor_op(data.data, fun, **kwargs)
                return data
    return data


def tensor_to(data, non_blocking=False, check_slowdown=False, **kwargs):
    def fun(data, check_slowdown, **kwargs):
        if check_slowdown:
            device = kwargs.get("device", None)
            non_blocking = kwargs.get("non_blocking", False)
            if (
                str(data.device) == "cpu"
                and device is not None
                and str(device) != str(data.device)
            ):
                if not data.is_pinned():
                    raise RuntimeError("tensor is not pinned")
                if not non_blocking:
                    raise RuntimeError(
                        "copy is blocking",
                    )
            else:
                if device is not None and not kwargs.get("non_blocking", False):
                    raise RuntimeError(
                        "device to device copy is blocking",
                    )
        return data.to(**kwargs)

    return recursive_tensor_op(
        data, fun, non_blocking=non_blocking, check_slowdown=check_slowdown, **kwargs
    )


def tensor_clone(data, detach=True):
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
