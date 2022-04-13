import functools

import torch
import torch.nn as nn
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
        self.__need_float_targets: bool = False

    @property
    def example_input(self):
        assert self.__example_input
        return self.__example_input

    @property
    def model(self) -> torch.nn.Module:
        return self.__model

    @property
    def model_util(self) -> ModelUtil:
        return ModelUtil(self.model)

    @property
    def has_batch_norm(self):
        if self.__has_batch_norm is None:
            self.__has_batch_norm = self.model_util.has_sub_module(torch.nn.BatchNorm2d)
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

    def offload_from_gpu(self):
        self.model.zero_grad(set_to_none=True)
        self.model.cpu()

    def __call__(
        self,
        inputs,
        targets,
        phase: MachineLearningPhase = None,
        device=None,
        non_blocking=False,
        batch_size=None,
        model_fun=None,
        parameters=None,
    ) -> dict:
        if model_fun is not None:
            if phase is not None:
                self.set_model_mode(phase == MachineLearningPhase.Training)

        multiple_input = isinstance(inputs, tuple | list)
        if device is not None:
            inputs = put_data_to_device(
                inputs, device=device, non_blocking=non_blocking
            )
            targets = put_data_to_device(
                targets, device=device, non_blocking=non_blocking
            )
            if model_fun is None:
                if next(self.model.parameters()).device != device:
                    self.model.to(device, non_blocking=non_blocking)
            else:
                parameters = tuple(
                    p.to(device, non_blocking=non_blocking) for p in parameters
                )

        assert self.loss_fun is not None
        if model_fun is not None:
            assert parameters is not None
            model = functools.partial(model_fun, parameters)
        else:
            model = self.model
        if not multiple_input:
            output = model(inputs)
        else:
            output = model(*inputs)
        if self.__need_float_targets:
            targets = targets.to(output.dtype, non_blocking=non_blocking)
        loss = self.loss_fun(output, targets)
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
            for _, m in self.model_util.get_sub_modules()
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
        if isinstance(self.__model, torch.quantization.stubs.QuantWrapper):
            return self.__model.module
        return self.__model

    def __is_averaged_loss(self) -> bool:
        if hasattr(self.loss_fun, "reduction"):
            if self.loss_fun.reduction in ("mean", "elementwise_mean"):
                return True
        return False

    def __repr__(self):
        return f"model: {self.__model.__class__.__name__}, loss_fun: {self.loss_fun}"

    def set_model_mode(self, is_training: bool) -> None:
        if is_training:
            if self.__model.training:
                return
            self.__model.train()
            return
        if self.__model.training:
            self.__model.eval()


class CheckPointedModelWithLoss(ModelWithLoss):
    """
    Combine a model with a loss function.
    """

    def __init__(self, model: torch.nn.Module, *args, **kwargs):
        super().__init__(get_checkpointed_model(model), *args, **kwargs)

    def __call__(self, inputs: torch.Tensor, *args, **kwargs) -> dict:
        if self.model.training:
            inputs.requires_grad_()
        return super().__call__(inputs, *args, **kwargs)
