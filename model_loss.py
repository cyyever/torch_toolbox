import torch
import torch.nn as nn
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN


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

    def __call__(self, inputs, target, for_training: bool) -> dict:
        if isinstance(self.__model, GeneralizedRCNN):
            if for_training:
                loss_dict: dict = self.__model(inputs, target)
                return {"loss": sum(loss for loss in loss_dict.values())}
            return self.__model(inputs)

        assert self.__loss_fun is not None

        output = self.__model(inputs)
        loss = self.__loss_fun(output, target)
        return {"loss": loss, "output": output}

    def __choose_loss_function(self) -> torch.nn.modules.loss._Loss:
        last_layer = list(self.__model.modules())[-1]
        if isinstance(last_layer, nn.LogSoftmax):
            return nn.NLLLoss()
        if isinstance(last_layer, nn.Linear):
            return nn.CrossEntropyLoss()
        raise NotImplementedError()

    def is_averaged_loss(self) -> bool:
        if hasattr(self.loss_fun, "reduction"):
            if self.loss_fun.reduction in ("mean", "elementwise_mean"):
                return True
        return False

    def __str__(self):
        return "model: {}, loss_fun: {}".format(
            self.model.__class__.__name__, self.loss_fun
        )
