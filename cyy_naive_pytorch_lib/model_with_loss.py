from typing import Optional

import torch
import torch.nn as nn
import torchvision
from cyy_naive_lib.log import get_logger
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN

from ml_type import MachineLearningPhase, ModelType
from model_util import ModelUtil


class ModelWithLoss:
    """
    Combine a model with a loss function.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fun=None,
        model_type: ModelType = None,
    ):
        self.__model = model
        self.__loss_fun: torch.nn.modules.loss._Loss = loss_fun
        if isinstance(loss_fun, str):
            if loss_fun == "CrossEntropyLoss":
                self.__loss_fun = nn.CrossEntropyLoss()
            else:
                raise RuntimeError("unknown loss function %s", loss_fun)
        self.__model_type = model_type
        self.__has_batch_norm = None
        self.__model_transforms: list = list()

    @property
    def model(self) -> torch.nn.Module:
        return self.__model

    @property
    def has_batch_norm(self):
        if self.__has_batch_norm is None:
            pass
        self.__has_batch_norm = ModelUtil(self.model).has_sub_module(
            torch.nn.BatchNorm2d
        )
        return self.__has_batch_norm

    @property
    def model_type(self):
        return self.__model_type

    @property
    def loss_fun(self) -> torch.nn.modules.loss._Loss:
        if self.__loss_fun is None:
            self.__loss_fun = self.__choose_loss_function()
        return self.__loss_fun

    def set_loss_fun(self, loss_fun: torch.nn.modules.loss._Loss):
        self.__loss_fun = loss_fun

    def append_transform(self, transform):
        self.__model_transforms.append(transform)

    def set_model(self, model: torch.nn.Module):
        self.__model = model

    def __call__(
        self, inputs, targets, phase: MachineLearningPhase = None, device=None
    ) -> dict:
        if phase is not None:
            self.__set_model_mode(phase)
        else:
            if self.model.training:
                phase = MachineLearningPhase.Training
        if device is not None:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            self.model.to(device, non_blocking=True)

        assert self.loss_fun is not None
        if self.__model_transforms and isinstance(self.__model_transforms, list):
            self.__model_transforms = torchvision.transforms.Compose(
                self.__model_transforms
            )
        if not isinstance(self.__model_transforms, list):
            inputs = self.__model_transforms(inputs)
        output = self.__model(inputs)
        loss = self.loss_fun(output, targets)
        return {"loss": loss, "output": output}

    def __choose_loss_function(self) -> Optional[torch.nn.modules.loss._Loss]:
        if isinstance(self.__model, GeneralizedRCNN):
            return None
        layers = [
            m
            for m in self.__model.modules()
            if not isinstance(
                m, (torch.quantization.QuantStub, torch.quantization.DeQuantStub)
            )
        ]
        last_layer = layers[-1]

        if isinstance(last_layer, nn.LogSoftmax):
            return nn.NLLLoss()
        if isinstance(last_layer, nn.Linear):
            return nn.CrossEntropyLoss()
        get_logger().error("can't choose a loss function, layers are %s", layers)
        raise NotImplementedError(type(last_layer))

    def is_averaged_loss(self) -> bool:
        if hasattr(self.loss_fun, "reduction"):
            if self.loss_fun.reduction in ("mean", "elementwise_mean"):
                return True
        return False

    def __str__(self):
        return "model: {}, loss_fun: {}".format(
            self.model.__class__.__name__, self.loss_fun
        )

    def __set_model_mode(self, phase: MachineLearningPhase):
        if phase == MachineLearningPhase.Training:
            self.model.train()
            return
        self.model.eval()
