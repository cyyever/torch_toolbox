import functools
from typing import Any, Callable

from cyy_naive_lib.log import get_logger

from ..dependency import has_torch_geometric

if has_torch_geometric:
    from .graph import pyg_data_extraction


def default_data_extraction(data: Any, extract_index: bool = True) -> dict:
    if has_torch_geometric:
        result = pyg_data_extraction(data=data, extract_index=extract_index)
        if result is not None:
            return result
    if extract_index:
        match data:
            case {"data": real_data, "index": index}:
                return default_data_extraction(real_data, extract_index=False) | {
                    "index": index
                }
            case [index, real_data]:
                return default_data_extraction(real_data, extract_index=False) | {
                    "index": index
                }
            case {"index": index, **real_data}:
                return default_data_extraction(real_data, extract_index=False) | {
                    "index": index
                }
            case {"input": _, "edge_index": __, **___}:
                return data
        raise NotImplementedError(data)
    match data:
        case [sample_input, target]:
            return {"input": sample_input, "target": target}
        case {"label": label, **other_data}:
            return {"target": label, "input": other_data}
        case _:
            return data


def __get_int_target(
    reversed_label_names: dict, label_name: str, *args: list, **kwargs: dict
) -> int:
    return reversed_label_names[label_name]


def str_target_to_int(label_names: dict) -> Callable:
    reversed_label_names = {v: k for k, v in label_names.items()}
    get_logger().info("map string targets by %s", reversed_label_names)
    return functools.partial(__get_int_target, reversed_label_names)


def int_target_to_text(target, index):
    match target:
        case 0:
            return "zero"
        case 1:
            return "one"
        case 2:
            return "two"
        case 3:
            return "three"
        case int():
            raise NotImplementedError()
    return target


def __replace_target(label_map, target, index):
    if index in label_map:
        assert label_map[index] != target
        target = label_map[index]
    return target


def replace_target(label_map: dict) -> Callable:
    return functools.partial(__replace_target, label_map)


def swap_input_and_target(data):
    data["input"], data["target"] = data["target"], data["input"]
    return data


def replace_str(string, old, new):
    return string.replace(old, new)


def target_offset(target, index: int, offset: int):
    return target + offset
