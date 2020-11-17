import torch
import torch.nn as nn


def cat_tensors_to_vector(tensors):
    return nn.utils.parameters_to_vector([t.reshape(-1) for t in tensors])


def get_batch_size(tensors):
    if isinstance(tensors, torch.Tensor):
        return tensors.shape[0]
    if isinstance(tensors, list):
        return len(tensors)
    raise RuntimeError("invalid tensors:" + str(tensors))
