import functools
from typing import Any, Callable

from cyy_naive_lib.log import get_logger

from ..dependency import has_torch_geometric


def default_data_extraction(data: Any, extract_index: bool = True) -> dict:
    if has_torch_geometric:
        match data:
            case {
                "subset_mask": subset_mask,
                "graph": graph,
            }:
                res = {
                    "input": {"x": graph.x, "edge_index": graph.edge_index},
                    "target": graph.y,
                    "mask": subset_mask,
                }
                return res
            # case torch_geometric.data.Data():
            #     res = {
            #         "input": {"x": data.x, "edge_index": data.edge_index},
            #         "target": data.y,
            #     }
            #     for attr_name in ["mask"]:
            #         if hasattr(data, attr_name):
            #             res[attr_name] = getattr(data, attr_name)
            #     return res
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
            case {"input": sample_input, "target": target, "index": index}:
                return data
            case {"target": target, "index": index}:
                return data
            case dict():
                if "input" in data and "edge_index" in data["input"]:
                    return data
        raise NotImplementedError(data)
    match data:
        case [sample_input, target]:
            return {"input": sample_input, "target": target}
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
