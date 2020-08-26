from typing import Callable
from cyy_naive_lib.list_op import split_list_to_chunks


def iterate_over_synced_tensor_dict(tensor_dict, cb: Callable):
    keys = set(tensor_dict.keys())
    in_memory_keys = set(tensor_dict.in_memory_keys())
    for k in in_memory_keys:
        cb(k, tensor_dict[k])
    remain_keys = keys - in_memory_keys
    cache_size = tensor_dict.get_in_memory_number()
    for chunk in split_list_to_chunks(remain_keys, cache_size // 2):
        tensor_dict.prefetch(chunk)
        for k in chunk:
            cb(k, tensor_dict[k])
