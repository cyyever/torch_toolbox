from typing import Callable
from cyy_naive_lib.list_op import split_list_to_chunks


def iterate_over_synced_tensor_dict(dict, cb: Callable):
    keys = set(dict.keys())
    in_memory_keys = set(dict.in_memory_keys())
    for k in in_memory_keys:
        cb(k, dict[k])
    remain_keys = keys - in_memory_keys
    for chunk in split_list_to_chunks(remain_keys, 100):
        dict.prefetch(chunk)
        for k in chunk:
            cb(k, dict[k])

        # self.cache_size // 3):
