import numpy as np
import torch
import torch.nn as nn
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_order


def cat_tensors_to_vector(tensors) -> torch.Tensor:
    return nn.utils.parameters_to_vector([t.reshape(-1) for t in tensors])


def get_batch_size(tensors):
    if isinstance(tensors, torch.Tensor):
        return tensors.shape[0]
    if isinstance(tensors, list):
        return len(tensors)
    raise RuntimeError("invalid tensors:" + str(tensors))


class TensorUtil:
    def __init__(self, data):
        self.__data = data

    @property
    def data(self):
        return self.__data

    def concat_dict_values(self) -> torch.Tensor:
        assert isinstance(self.data, dict)
        return cat_tensors_to_vector(get_mapping_values_by_order(self.data))

    def load_dict_values(self, values: torch.Tensor):
        bias = 0
        for name in sorted(self.data.keys()):
            shape = self.__data[name].shape
            param_element_num = np.prod(shape)
            self.__data[name] = values.narrow(0, bias, param_element_num).view(*shape)
            bias += param_element_num
        assert bias == values.shape[0]
