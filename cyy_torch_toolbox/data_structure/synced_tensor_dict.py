from typing import Generator

from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_naive_lib.log import get_logger
from cyy_torch_cpp_extension.data_structure import \
    SyncedSparseTensorDict as SyncedSparseTensorDict__
from cyy_torch_cpp_extension.data_structure import \
    SyncedTensorDict as SyncedTensorDict__


class SyncedTensorDict:
    def __init__(self, tensor_dict, key_type):
        self.__tensor_dict = tensor_dict
        self.__key_type = key_type

    def __contains__(self, item):
        return self.__tensor_dict.__contains__(str(item))

    def __getitem__(self, key):
        return self.__tensor_dict.__getitem__(str(key))

    def __setitem__(self, key, value):
        self.__tensor_dict.__setitem__(str(key), value)

    def __delitem__(self, key):
        self.__tensor_dict.__delitem__(str(key))

    def release(self):
        self.__tensor_dict.release()

    def get_storage_dir(self):
        self.__tensor_dict.get_storage_dir()

    def keys(self) -> set:
        return {self.__key_type(k) for k in self.__tensor_dict.keys()}

    def in_memory_keys(self) -> set:
        return {self.__key_type(k) for k in self.__tensor_dict.in_memory_keys()}

    def prefetch(self, keys: set):
        self.__tensor_dict.prefetch([str(k) for k in keys])

    def __getattr__(self, name):
        return getattr(self.__tensor_dict, name)

    @property
    def tensor_dict(self):
        return self.__tensor_dict

    def iterate(self, keys: set = None) -> Generator:
        if keys is None:
            keys = set(self.__tensor_dict.keys())
        else:
            keys = {str(k) for k in keys}
        in_memory_keys = set(self.__tensor_dict.in_memory_keys()) & keys
        for k in in_memory_keys:
            yield (self.__key_type(k), self.__tensor_dict[k])
        remain_keys = list(keys - in_memory_keys)
        cache_size = self.__tensor_dict.get_in_memory_number()
        for chunk in split_list_to_chunks(remain_keys, cache_size // 2):
            self.__tensor_dict.prefetch(chunk)
            for k in chunk:
                yield (self.__key_type(k), self.__tensor_dict[k])

    @staticmethod
    def create(
        key_type=int,
        cache_size=None,
        mask=None,
        tensor_shape=None,
        storage_dir=None,
    ):
        if not storage_dir:
            storage_dir = ""
        if mask is not None:
            assert tensor_shape is not None
            m = SyncedSparseTensorDict__(mask, tensor_shape, storage_dir)
        else:
            m = SyncedTensorDict__(storage_dir)
        m.set_permanent_storage()
        if cache_size is not None:
            m.set_in_memory_number(cache_size)
        get_logger().info("tensor_dict use cache size %s", cache_size)
        m.set_logging(False)
        return SyncedTensorDict(tensor_dict=m, key_type=key_type)
