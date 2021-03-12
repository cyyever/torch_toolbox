from typing import Optional

import torch
import torch.nn as nn
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN

from ml_type import MachineLearningPhase, ModelType


class ModelWithLoss:
    """
    Combine a model with a loss function.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fun: torch.nn.modules.loss._Loss = None,
        model_type: ModelType = None,
    ):
        self.__model = model
        self.__loss_fun = loss_fun
        if self.__loss_fun is None:
            self.__loss_fun = self.__choose_loss_function()
        self.__model_type = model_type

    @property
    def model(self) -> torch.nn.Module:
        return self.__model

    @property
    def model_type(self):
        return self.__model_type

    @property
    def loss_fun(self):
        return self.__loss_fun

    def set_model(self, model: torch.nn.Module):
        self.__model = model

    def __call__(self, inputs, target, phase: MachineLearningPhase = None) -> dict:
        if phase is not None:
            self.__set_model_mode(phase)
        else:
            if self.model.training:
                phase = MachineLearningPhase.Training
        if isinstance(self.model, GeneralizedRCNN):
            detection = None
            assert phase is not None
            if phase == MachineLearningPhase.Training:
                loss_dict = self.model(inputs, target)
            else:
                loss_dict, detection = self.model(inputs, target)

            result = {"loss": sum(loss for loss in loss_dict.values())}
            if detection is not None:
                result["detection"] = detection
            return result

        assert self.__loss_fun is not None

        output = self.__model(inputs)
        loss = self.__loss_fun(output, target)
        return {"loss": loss, "output": output}

    def __choose_loss_function(self) -> Optional[torch.nn.modules.loss._Loss]:
        if isinstance(self.__model, GeneralizedRCNN):
            return None
        last_layer = [
            m
            for m in self.__model.modules()
            if not isinstance(
                m, (torch.quantization.QuantStub, torch.quantization.DeQuantStub)
            )
        ][-1]
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

    def __set_model_mode(self, phase: MachineLearningPhase):
        if isinstance(self.__model, GeneralizedRCNN):
            if phase == MachineLearningPhase.Training:
                self.model.train()
            else:
                self.model.eval()
            return

        if phase == MachineLearningPhase.Training:
            self.model.train()
            return
        self.model.eval()
