from typing import Callable

import torch
import torch.nn as nn
import transformers
from cyy_naive_lib.log import get_logger
from torch.nn.parallel import DistributedDataParallel as DDP

from device import get_devices, put_data_to_device
from ml_type import MachineLearningPhase, ModelType
from model_transformers.checkpointed_model import get_checkpointed_model
from model_util import ModelUtil

try:
    import apex

    has_apex = True
except ModuleNotFoundError:
    has_apex = False


class ModelWithLoss:
    """
    Combine a model with a loss function.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fun: str | Callable | None = None,
        model_type: ModelType = None,
    ):
        self._model: torch.nn.Module = model

        self.__loss_fun: Callable | None = None
        if loss_fun is not None:
            self.set_loss_fun(loss_fun)
        self.__model_type = model_type
        self.__has_batch_norm = None
        self.__example_input = None
        self.__need_float_targets: bool | None = None

    @property
    def example_input(self):
        assert self.__example_input
        return self.__example_input

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def model_util(self) -> ModelUtil:
        return ModelUtil(self.model)

    @property
    def has_batch_norm(self):
        if self.__has_batch_norm is None:
            self.__has_batch_norm = self.model_util.have_sub_module(
                sub_module_type=torch.nn.BatchNorm2d
            )
        return self.__has_batch_norm

    @property
    def model_type(self):
        return self.__model_type

    @property
    def loss_fun(self) -> Callable:
        if self.__loss_fun is None:
            self.__loss_fun = self.__choose_loss_function()
        return self.__loss_fun

    def set_loss_fun(self, loss_fun: Callable | str) -> None:
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

    @property
    def need_float_targets(self):
        if self.__need_float_targets is None:
            self.__need_float_targets = False
            if isinstance(self.__loss_fun, nn.BCEWithLogitsLoss):
                self.__need_float_targets = True
        return self.__need_float_targets

    def __call__(
        self,
        inputs,
        targets,
        phase: MachineLearningPhase = None,
        device=None,
        non_blocking=False,
        input_features=None,
    ) -> dict:
        if phase is not None:
            self.set_model_mode(phase == MachineLearningPhase.Training)

        if device is not None:
            if input_features is not None:
                input_features = put_data_to_device(
                    input_features, device=device, non_blocking=non_blocking
                )
            else:
                inputs = put_data_to_device(
                    inputs, device=device, non_blocking=non_blocking
                )
            targets = put_data_to_device(
                targets, device=device, non_blocking=non_blocking
            )
            self.model.to(device, non_blocking=non_blocking)

        model = self.model
        if hasattr(model, "forward_input_feature"):
            if input_features is None:
                input_features = model.get_input_feature(inputs)
            model = model.forward_input_feature
        if input_features is not None:
            real_inputs = input_features
        else:
            real_inputs = inputs
        match real_inputs:
            case tuple():
                output = model(*real_inputs)
            case transformers.tokenization_utils_base.BatchEncoding():
                output = model(**real_inputs, labels=targets)
            case torch.Tensor():
                output = model(real_inputs)
            case _:
                raise NotImplementedError()
        loss = None
        match real_inputs:
            case transformers.tokenization_utils_base.BatchEncoding():
                loss = output["loss"]
                classification_output = output["logits"]
            case _:
                if self.need_float_targets:
                    targets = targets.to(dtype=output.dtype, non_blocking=non_blocking)
                assert self.loss_fun is not None
                loss = self.loss_fun(output, targets)
                classification_output = output
        is_averaged_loss = self.__is_averaged_loss()
        if is_averaged_loss is None:
            is_averaged_loss = classification_output is not None
        return {
            "loss": loss,
            "classification_output": classification_output,
            "inputs": inputs,
            "input_features": input_features,
            "targets": targets,
            "is_averaged_loss": is_averaged_loss,
        }

    def replace_model(self, model):
        return ModelWithLoss(
            model=model, loss_fun=self.loss_fun, model_type=self.model_type
        )

    def to(self, device, non_blocking=False):
        try:
            param = next(self.model.parameters())
        except StopIteration:
            param = next(self.model.buffers())
        if param.device != device:
            self.model.to(device, non_blocking=non_blocking)

    def __choose_loss_function(self) -> torch.nn.modules.loss._Loss:
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
            get_logger().debug("choose loss function NLLLoss")
            return nn.NLLLoss()
        if isinstance(last_layer, nn.Linear):
            if last_layer.out_features == 1:
                get_logger().debug("choose loss function BCEWithLogitsLoss")
                return nn.BCEWithLogitsLoss()
            get_logger().debug("choose loss function CrossEntropyLoss")
            return nn.CrossEntropyLoss()
        get_logger().error("can't choose a loss function, model is %s", self._model)
        raise NotImplementedError(type(last_layer))

    def get_real_model(self):
        if isinstance(self._model, torch.quantization.stubs.QuantWrapper):
            return self._model.module
        return self._model

    def __is_averaged_loss(self) -> bool | None:
        if hasattr(self.loss_fun, "reduction"):
            if self.loss_fun.reduction in ("mean", "elementwise_mean"):
                return True
            return False
        return None

    def __repr__(self):
        return f"model: {self._model.__class__.__name__}, loss_fun: {self.loss_fun}"

    def set_model_mode(self, is_training: bool) -> None:
        if is_training:
            if self._model.training:
                return
            self._model.train()
            return
        if self._model.training:
            self._model.eval()


class CheckPointedModelWithLoss(ModelWithLoss):
    """
    Combine a model with a loss function.
    """

    def __init__(self, model: torch.nn.Module, *args, **kwargs):
        super().__init__(get_checkpointed_model(model), *args, **kwargs)

    def __call__(self, **kwargs) -> dict:
        if self.model.training:
            inputs = kwargs.get("inputs", None)
            if inputs is not None:
                inputs.requires_grad_()
            input_features = kwargs.get("input_features", None)
            if input_features is not None:
                input_features.requires_grad_()
        return super().__call__(**kwargs)


class ParallelModelWithLoss(ModelWithLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert torch.cuda.is_available()
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="tcp://127.0.0.1:23456",
                rank=0,
                world_size=len(get_devices()),
            )
        self._original_model = self._model
        self._model = DDP(self._original_model)

    @classmethod
    def create(cls, model_with_loss: ModelWithLoss):
        return cls(
            model=model_with_loss.model,
            loss_fun=model_with_loss.loss_fun,
            model_type=model_with_loss.model_type,
        )
