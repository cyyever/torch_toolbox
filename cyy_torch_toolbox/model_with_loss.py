import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torchvision
from cyy_naive_lib.log import get_logger

from ml_type import MachineLearningPhase, ModelType
from model_util import ModelUtil


class CheckPointBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = nn.Sequential(*[m[1] for m in block])
        self.__block_names = [m[0] for m in block]
        get_logger().debug("use checkpoint_block %s", self.__block_names)

    def forward(self, x):
        return self.block(x)


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
                raise RuntimeError("unknown loss function {}".format(loss_fun))
        self.__model_type = model_type
        self.__has_batch_norm = None
        self.__model_transforms: list = list()
        self.trace_input = False
        self.__example_input = None
        self.use_checkpoint = False
        self.__checkpointed_model = None
        self.__current_phase = None
        self.__current_model_device = None

    @property
    def example_input(self):
        assert self.__example_input
        return self.__example_input

    @property
    def model(self) -> torch.nn.Module:
        self.__current_phase = None
        self.__current_model_device = None
        return self.__model

    @property
    def has_batch_norm(self):
        if self.__has_batch_norm is None:
            pass
        self.__has_batch_norm = ModelUtil(self.__model).has_sub_module(
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


    @property
    def checkpointed_model(self) -> torch.nn.Module:
        if self.__checkpointed_model is not None:
            return self.__checkpointed_model
        checkpointed_blocks = ModelUtil(self.__model).get_sub_module_blocks(
            block_types={(nn.Conv2d, nn.BatchNorm2d)},
            # block_types={},
            only_block_name=False,
        )
        assert checkpointed_blocks
        self.__checkpointed_model = copy.deepcopy(self.__model)
        self.__checkpointed_model.load_state_dict(self.__model.state_dict())
        checkpointed_model_util = ModelUtil(self.__checkpointed_model)
        for checkpointed_block in checkpointed_blocks:
            for idx, submodule in enumerate(checkpointed_block):
                submodule_name = submodule[0]
                if idx == 0:
                    checkpointed_model_util.set_attr(
                        submodule_name,
                        CheckPointBlock(checkpointed_block),
                        as_parameter=False,
                    )
                else:
                    checkpointed_model_util.set_attr(
                        submodule_name, lambda x: x, as_parameter=False
                    )

        return self.__checkpointed_model

    def __call__(
        self,
        inputs,
        targets,
        phase: MachineLearningPhase = None,
        device=None,
        non_blocking=False,
    ) -> dict:
        if phase is not None:
            if self.__current_phase != phase:
                self.__set_model_mode(phase)
            self.__current_phase = phase
        else:
            if self.__model.training:
                self.__current_phase = MachineLearningPhase.Training
            else:
                self.__current_phase = None

        extra_inputs = []
        if isinstance(inputs, tuple):
            inputs, *extra_inputs = inputs

        if device is not None:
            inputs = inputs.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            if self.__current_model_device != device:
                self.__model.to(device, non_blocking=non_blocking)
                self.__current_model_device = device

        assert self.loss_fun is not None
        if self.__model_transforms:
            if isinstance(self.__model_transforms, list):
                self.__model_transforms = torchvision.transforms.Compose(
                    self.__model_transforms
                )
            inputs = self.__model_transforms(inputs)
        if (
            self.__current_phase == MachineLearningPhase.Training
            and self.use_checkpoint
        ):
            inputs.requires_grad_()
            output = self.checkpointed_model(inputs, *extra_inputs)
        else:
            output = self.__model(inputs, *extra_inputs)
        loss = self.loss_fun(output, targets)
        if self.trace_input and self.__example_input is None:
            self.__example_input = [inputs.detach()] + copy.deepcopy(extra_inputs)
        normalized_loss = loss
        if self.__is_averaged_loss():
            normalized_loss = loss * targets.shape[0]
        return {"loss": loss, "normalized_loss": normalized_loss, "output": output}

    def __choose_loss_function(self) -> Optional[torch.nn.modules.loss._Loss]:
        # if isinstance(self.__model, GeneralizedRCNN):
        #     return None
        layers = [
            m
            for _, m in ModelUtil(self.__model).get_sub_modules()
            if not isinstance(
                m,
                (
                    torch.quantization.QuantStub,
                    torch.quantization.DeQuantStub,
                    torch.quantization.stubs.QuantWrapper,
                    torch.quantization.fake_quantize.FakeQuantize,
                    torch.quantization.observer.MovingAverageMinMaxObserver,
                    torch.quantization.observer.MovingAveragePerChannelMinMaxObserver,
                    torch.nn.modules.dropout.Dropout,
                ),
            )
            and "MemoryEfficientSwish" not in str(m)
        ]
        last_layer = layers[-1]

        get_logger().info("last module is %s", last_layer.__class__)
        if isinstance(last_layer, nn.LogSoftmax):
            return nn.NLLLoss()
        if isinstance(last_layer, nn.Linear):
            return nn.CrossEntropyLoss()
        get_logger().error("can't choose a loss function, model is %s", self.__model)
        raise NotImplementedError(type(last_layer))

    def __is_averaged_loss(self) -> bool:
        if hasattr(self.loss_fun, "reduction"):
            if self.loss_fun.reduction in ("mean", "elementwise_mean"):
                return True
        return False

    def __str__(self):
        return "model: {}, loss_fun: {}".format(
            self.__model.__class__.__name__, self.loss_fun
        )

    def __set_model_mode(self, phase: MachineLearningPhase):
        if phase == MachineLearningPhase.Training:
            self.__model.train()
            return
        self.__model.eval()
