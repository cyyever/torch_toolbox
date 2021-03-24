from typing import Generator

import cyy_naive_cpp_extension.data_structure.SyncedSparseTensorDict as __SyncedSparseTensorDict
import cyy_naive_cpp_extension.data_structure.SyncedTensorDict as __SyncedTensorDict
import torch
import torch.nn.utils.prune as prune
from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_naive_lib.log import get_logger

from model_util import ModelUtil


class SyncedTensorDict:
    def __init__(self, tensor_dict):
        self.__tensor_dict = tensor_dict

    def iterate(self, keys: set = None) -> Generator:
        if keys is None:
            keys = set(self.__tensor_dict.keys())
        else:
            keys = {str(k) for k in keys}
        in_memory_keys = set(self.__tensor_dict.in_memory_keys()) & keys
        for k in in_memory_keys:
            yield (k, self.__tensor_dict[k])
        remain_keys = list(keys - in_memory_keys)
        cache_size = self.__tensor_dict.get_in_memory_number()
        for chunk in split_list_to_chunks(remain_keys, cache_size // 2):
            self.__tensor_dict.prefetch(chunk)
            for k in chunk:
                yield (k, self.__tensor_dict[k])

    @staticmethod
    def create(
        cache_size,
        mask=None,
        tensor_shape=None,
        storage_dir=None,
    ):
        if not storage_dir:
            storage_dir = ""
        if mask is not None:
            assert tensor_shape is not None
            m = __SyncedSparseTensorDict(mask, tensor_shape, storage_dir)
        else:
            m = __SyncedTensorDict(storage_dir)
        m.set_permanent_storage()
        m.set_in_memory_number(cache_size)
        get_logger().info("tensor_dict use cache size %s", cache_size)
        m.set_logging(False)
        return SyncedTensorDict(m)
