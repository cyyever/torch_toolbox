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


    def prefetch(self, keys: set):
        self.__tensor_dict.prefetch(self.__change_key_type(keys))

    @staticmethod
    def __change_key_type(keys):
        return {str(k) for k in keys}

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


def iterate_over_synced_tensor_dict(
    tensor_dict: SyncedTensorDict,
    keys: set = None,
) -> Generator:
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


def create_tensor_dict(
    cache_size,
    model=None,
    storage_dir=None,
    concat_momentum=False,
):
    if not storage_dir:
        storage_dir = ""
    mask = None
    gradient_shape = None
    if model is not None and prune.is_pruned(model):
        model_util = ModelUtil(model)
        get_logger().info(
            "use pruned model, sparsity is %s", model_util.get_sparsity()[0]
        )
        parameters = model_util.get_parameter_list()
        gradient_shape = parameters.shape
        mask = model_util.get_pruning_mask_list()
        assert len(mask) == len(parameters)
    m = None
    if mask is not None:
        if concat_momentum:
            mask = torch.cat((mask, mask))
            gradient_shape[1] *= 2
        m = __SyncedSparseTensorDict(mask, gradient_shape, storage_dir)
    else:
        m = __SyncedTensorDict(storage_dir)
    m.set_permanent_storage()
    m.set_in_memory_number(cache_size)
    get_logger().info("gradient matrix use cache size %s", cache_size)
    m.set_logging(False)
    return m
