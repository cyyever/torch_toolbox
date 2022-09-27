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
    return nn.utils.parameters_to_vector([t.view(-1) for t in tensors])


class __RecursiveCheckPoint:
    def __init__(self, data):
        self.data = data


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


# def split_tensor_to_dict(name_and_shapes: list, tensor: torch.Tensor) -> dict:
#     data = {}
#     bias = 0
#     for (name, shape) in name_and_shapes:
#         param_element_num = numpy.prod(shape)
#         data[name] = tensor.narrow(0, bias, param_element_num).view(*shape)
#         bias += param_element_num
#     assert bias == tensor.shape[0]
#     return data


# def split_tensor_to_list(shapes: list, tensor: torch.Tensor) -> list:
#     data = []
#     bias = 0
#     for shape in shapes:
#         param_element_num = numpy.prod(shape)
#         data.append(tensor.narrow(0, bias, param_element_num).view(*shape))
#         bias += param_element_num
#     assert bias == tensor.shape[0]
#     return data


def get_tensor_serialization_size(data):
    return len(pickle.dumps(data))


def __recursive_tensor_op(data, fun, **kwargs) -> Any:
    match data:
        case __RecursiveCheckPoint():
            return fun(data.data, **kwargs)
        case torch.Tensor():
            return fun(data, **kwargs)
        case list():
            return [__recursive_tensor_op(element, fun, **kwargs) for element in data]
        case tuple():
            return tuple(__recursive_tensor_op(list(data), fun, **kwargs))
        case dict():
            # we need to check in key order because that order may be useful
            return {
                k: __recursive_tensor_op(data[k], fun, **kwargs)
                for k in sorted(data.keys())
            }
        case functools.partial():
            return functools.partial(
                data.func,
                *__recursive_tensor_op(data.args, fun, **kwargs),
                **__recursive_tensor_op(data.keywords, fun, **kwargs)
            )
    if has_hugging_face:
        match data:
            case transformers.tokenization_utils_base.BatchEncoding():
                data.data = __recursive_tensor_op(data.data, fun, **kwargs)
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
                and device != data.device
            ):
                if not data.is_pinned():
                    raise RuntimeError("tensor is not pinned")
                if not non_blocking:
                    raise RuntimeError(
                        "cpu to device copy is blocking",
                    )
            else:
                if device is not None and not kwargs.get("non_blocking", False):
                    raise RuntimeError(
                        "device to device copy is blocking",
                    )
        return data.to(**kwargs)

    return __recursive_tensor_op(
        data, fun, non_blocking=non_blocking, check_slowdown=check_slowdown, **kwargs
    )


def tensor_clone(data, detach=True):
    def fun(data, detach):
        new_data = data.clone()
        if detach:
            new_data = new_data.detach()
        return new_data

    return __recursive_tensor_op(data, fun, detach=detach)


def assemble_tensors(data: Any) -> tuple[torch.Tensor, Any]:
    single_tensor = []
    offset = 0

    def fun(data: Any) -> __RecursiveCheckPoint:
        nonlocal offset
        shape = data.shape
        old_offset = offset
        single_tensor.append(data.view(-1))
        offset += data.numel()
        return __RecursiveCheckPoint(data=(shape, old_offset))

    res = __recursive_tensor_op(data, fun)
    if offset == 0:
        assert not single_tensor
    if single_tensor:
        assert offset != 0
        return cat_tensors_to_vector(single_tensor), res
    return None, res


def disassemble_tensor(single_tensor, data: Any, clone=True) -> Any:
    def fun(data) -> torch.Tensor:
        shape, offset = data
        tensor = single_tensor[offset: offset + numpy.prod(shape)].view(*shape)
        if clone:
            tensor = tensor.clone()
        return tensor

    return __recursive_tensor_op(data, fun)
