from cyy_naive_lib.list_op import split_list_to_chunks


def iterate_over_synced_tensor_dict(tensor_dict, keys: set = None):
    if keys is None:
        keys = set(tensor_dict.keys())
    else:
        keys = {str(k) for k in keys}
    in_memory_keys = set(tensor_dict.in_memory_keys()) & keys
    for k in in_memory_keys:
        yield (k, tensor_dict[k])
    remain_keys = list(keys - in_memory_keys)
    cache_size = tensor_dict.get_in_memory_number()
    for chunk in split_list_to_chunks(remain_keys, cache_size // 2):
        tensor_dict.prefetch(chunk)
        for k in chunk:
            yield (k, tensor_dict[k])
