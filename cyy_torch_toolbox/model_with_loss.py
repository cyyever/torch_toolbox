# import copy

import torch
import torch.nn as nn
import torch.utils.checkpoint
from cyy_naive_lib.log import get_logger

from device import put_data_to_device
from ml_type import MachineLearningPhase, ModelType
from model_transformers.checkpointed_model import get_checkpointed_model
from model_util import ModelUtil


class ModelWithLoss:
    """
    Combine a model with a loss function.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fun: str = None,
        model_type: ModelType = None,
    ):
        self.__model: torch.nn.Module = model

        self.__loss_fun: torch.nn.modules.loss._Loss | None = None
        self.set_loss_fun(loss_fun)
        self.__model_type = model_type
        self.__has_batch_norm = None
        # self.__trace_input = False
        self.__example_input = None
        self.use_checkpointing = False
        self.__checkpointed_model: None | torch.nn.Module = None
        self.__model_in_trainig_mode: None | bool = None
        self.__current_model_device = None
        self.__need_float_targets: bool = False

    @property
    def example_input(self):
        assert self.__example_input
        return self.__example_input

    @property
    def model(self) -> torch.nn.Module:
        self.__current_model_device = None
        return self.__model

    @property
    def model_util(self) -> ModelUtil:
        return ModelUtil(self.__model)

    @property
    def has_batch_norm(self):
        if self.__has_batch_norm is None:
            self.__has_batch_norm = ModelUtil(self.get_real_model()).has_sub_module(
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

    def set_loss_fun(self, loss_fun: torch.nn.modules.loss._Loss | str | None) -> None:
        if isinstance(loss_fun, str):
            if loss_fun == "CrossEntropyLoss":
                self.__loss_fun = nn.CrossEntropyLoss()
            else:
                raise RuntimeError(f"unknown loss function {loss_fun}")
        else:
            self.__loss_fun = loss_fun

    @property
    def checkpointed_model(self) -> torch.nn.Module:
        if self.__checkpointed_model is not None:
            return self.__checkpointed_model
        self.__checkpointed_model = get_checkpointed_model(self.__model)
        return self.__checkpointed_model

    def offload_from_gpu(self):
        self.model.zero_grad(set_to_none=True)
        self.model.cpu()
        self.__checkpointed_model = None

    def __call__(
        self,
        inputs,
        targets,
        phase: MachineLearningPhase = None,
        device=None,
        non_blocking=False,
        batch_size=None,
    ) -> dict:
        if phase is not None:
            self.set_model_mode(phase)
        else:
            if self.__model.training:
                self.__model_in_trainig_mode = True
            else:
                self.__model_in_trainig_mode = False

        multiple_input = isinstance(inputs, tuple | list)
        if device is not None:
            inputs = put_data_to_device(
                inputs, device=device, non_blocking=non_blocking
            )
            targets = put_data_to_device(
                targets, device=device, non_blocking=non_blocking
            )
            if self.__current_model_device != device:
                self.__model.to(device, non_blocking=non_blocking)
                self.__current_model_device = device

        assert self.loss_fun is not None
        if self.__model_in_trainig_mode and self.use_checkpointing:
            if not multiple_input:
                inputs.requires_grad_()
            model = self.checkpointed_model
        else:
            model = self.__model
        if not multiple_input:
            output = model(inputs)
        else:
            output = model(*inputs)
        if self.__need_float_targets:
            targets = targets.to(output.dtype, non_blocking=non_blocking)
        loss = self.loss_fun(output, targets)
        # if self.__trace_input and self.__example_input is None:
        #     self.__example_input = [inputs.detach()] + copy.deepcopy(extra_inputs)
        normalized_loss = loss
        if batch_size is not None and self.__is_averaged_loss():
            normalized_loss = loss * batch_size
        return {
            "loss": loss,
            "normalized_loss": normalized_loss,
            "output": output,
            "inputs": inputs,
            "targets": targets,
        }

    def __choose_loss_function(self) -> torch.nn.modules.loss._Loss:
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

        get_logger().debug("last module is %s", last_layer.__class__)
        if isinstance(last_layer, nn.LogSoftmax):
            get_logger().warning("choose loss function NLLLoss")
            return nn.NLLLoss()
        if isinstance(last_layer, nn.Linear):
            if last_layer.out_features == 1:
                get_logger().warning("choose loss function BCEWithLogitsLoss")
                self.__need_float_targets = True
                return nn.BCEWithLogitsLoss()
            get_logger().warning("choose loss function CrossEntropyLoss")
            return nn.CrossEntropyLoss()
        get_logger().error("can't choose a loss function, model is %s", self.__model)
        raise NotImplementedError(type(last_layer))

    def get_real_model(self):
        if isinstance(self.model, torch.quantization.stubs.QuantWrapper):
            return self.model.module
        return self.model

    def __is_averaged_loss(self) -> bool:
        if hasattr(self.loss_fun, "reduction"):
            if self.loss_fun.reduction in ("mean", "elementwise_mean"):
                return True
        return False

    def __repr__(self):
        return f"model: {self.__model.__class__.__name__}, loss_fun: {self.loss_fun}"

    def set_model_mode(self, phase: MachineLearningPhase) -> None:
        if phase == MachineLearningPhase.Training:
            if self.__model_in_trainig_mode:
                return
            self.__model.train()
            self.__model_in_trainig_mode = True
            return
        if self.__model_in_trainig_mode is None:
            self.__model.eval()
            self.__model_in_trainig_mode = False
