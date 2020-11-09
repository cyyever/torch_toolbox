import torch
import torch.nn as nn


class ModelWithLoss:
    def __init__(
            self,
            model: torch.nn.Module,
            loss_fun: torch.nn.modules.loss._Loss = None):
        self.__model = model
        self.__loss_fun = loss_fun
        if self.__loss_fun is None:
            self.__loss_fun = self.__choose_loss_function()

    def set_model(self, model: torch.nn.Module):
        self.__model = model

    @property
    def model(self):
        return self.__model

    @property
    def loss_fun(self):
        return self.__loss_fun

    def __call__(self, tensor: torch.Tensor, target: torch.Tensor):
        assert self.__loss_fun is not None
        return self.__loss_fun(self.__model(tensor), target)

    def __choose_loss_function(self) -> torch.nn.modules.loss._Loss:
        last_layer = list(self.__model.modules())[-1]
        if isinstance(last_layer, nn.LogSoftmax):
            return nn.NLLLoss()
        if isinstance(last_layer, nn.Linear):
            return nn.CrossEntropyLoss()
        raise NotImplementedError()
