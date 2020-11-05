import torch.nn as nn


def cat_tensors_to_vector(tensors):
    return nn.utils.parameters_to_vector([t.reshape(-1) for t in tensors])
