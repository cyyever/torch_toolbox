import functools
from typing import Any, Callable

from cyy_naive_lib.log import log_info


def default_data_extraction(data: Any) -> dict:
    index = None
    match data:
        case {"data": data, "index": index}:
            pass
        case [index, data]:
            pass
        case {"index": index, **data}:
            pass
    match data:
        case [sample_input, target]:
            data = {"input": sample_input, "target": target}
        case {"label": label, "text": text}:
            data = {"target": label, "input": text}
        case {"label": label, **other_data}:
            data = {"target": label, "input": other_data}
    if index is not None:
        data["index"] = index
    return data


def __get_int_target(
    reversed_label_names: dict, label_name: str, *args: list, **kwargs: Any
) -> int:
    return reversed_label_names[label_name]


def str_target_to_int(label_names: dict) -> Callable:
    reversed_label_names = {v: k for k, v in label_names.items()}
    log_info("map string targets by %s", reversed_label_names)
    return functools.partial(__get_int_target, reversed_label_names)


def int_target_to_text(target: int, index: int, mapping: dict | None = None) -> str:
    if mapping is not None:
        return mapping[target]
    match target:
        case 0:
            return "zero"
        case 1:
            return "one"
        case 2:
            return "two"
        case 3:
            return "three"
    raise NotImplementedError()


def __replace_target(label_map, target, index):
    if index in label_map:
        assert label_map[index] != target
        target = label_map[index]
    return target


def replace_target(label_map: dict) -> Callable:
    return functools.partial(__replace_target, label_map)


def backup_target(data: dict) -> dict:
    data["original_target"] = data["target"]
    return data


def replace_str(string: str, old: str, new: str) -> str:
    return string.replace(old, new)


def target_offset(data: dict, offset: int) -> dict:
    data["target"] = data["target"] + offset
    return data
