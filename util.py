import random
import torch.nn as nn
import torch
import numpy as np

from cyy_naive_lib.log import get_logger


def set_reproducing_seed(reproducing_seed):
    get_logger().warning("set reproducing seed")
    assert isinstance(reproducing_seed, int)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(reproducing_seed)
    random.seed(reproducing_seed)
    np.random.seed(reproducing_seed)


def cat_tensors_to_vector(tensors):
    return nn.utils.parameters_to_vector([t.reshape(-1) for t in tensors])
