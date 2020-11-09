import torch
import torch.nn as nn


class ModelWithLoss:
    def __init__(
            self,
            model: torch.nn.Module,
            loss_fun: torch.nn.modules.loss._Loss = None):
        self.model = model
        self.loss_fun = loss_fun
        if self.loss_fun is None:
            self.loss_fun = self.__choose_loss_function()

    def __choose_loss_function(self) -> torch.nn.modules.loss._Loss:
        last_layer = list(self.model.modules())[-1]
        if isinstance(last_layer, nn.LogSoftmax):
            return nn.NLLLoss()
        if isinstance(last_layer, nn.Linear):
            return nn.CrossEntropyLoss()
        raise NotImplementedError()
