from typing import Generator
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.sequence_op import split_list_to_chunks
import cyy_pytorch_cpp

import torch
import torch.nn.utils.prune as prune
from model_util import ModelUtil


def iterate_over_synced_tensor_dict(
        tensor_dict: cyy_pytorch_cpp.data_structure.SyncedTensorDict,
        keys: set = None) -> Generator:
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
        m = cyy_pytorch_cpp.data_structure.SyncedSparseTensorDict(
            mask, gradient_shape, storage_dir
        )
    else:
        m = cyy_pytorch_cpp.data_structure.SyncedTensorDict(storage_dir)
    m.set_permanent_storage()
    m.set_in_memory_number(cache_size)
    get_logger().info("gradient matrix use cache size %s", cache_size)
    m.set_logging(False)
    return m
