import functools
from typing import Any, Callable

import torch
from cyy_naive_lib.log import get_logger
from torch import nn

from ..ml_type import MachineLearningPhase, ModelType
from ..model_util import ModelUtil
from ..tensor import tensor_to

# from cyy_torch_toolbox.model_transform.checkpointed_model import \
#     get_checkpointed_model
# from cyy_torch_toolbox.device import get_devices
# from torch.nn.parallel import DistributedDataParallel as DDP


class ModelEvaluator:
    """
    Combine a model with a loss function.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str | None = None,
        model_type: None | ModelType = None,
        loss_fun: str | Callable | None = None,
    ):
        self._model: torch.nn.Module = model
        self.__name = model_name
        self.__loss_fun: Callable | None = None
        if loss_fun is not None:
            self.set_loss_fun(loss_fun)
        if model_type is None:
            model_type = ModelType.Classification
        self.__model_type = model_type
        self.need_input_features = False

    @property
    def model_name(self) -> str | None:
        return self.__name

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @functools.cached_property
    def model_util(self) -> ModelUtil:
        return ModelUtil(self.model)

    @property
    def model_type(self) -> ModelType | None:
        return self.__model_type

    @property
    def loss_fun(self) -> Callable:
        if self.__loss_fun is None:
            self.__loss_fun = self.__choose_loss_function()
        return self.__loss_fun

    def set_loss_fun(self, loss_fun: Callable | str) -> None:
        match loss_fun:
            case "CrossEntropyLoss":
                self.__loss_fun = nn.CrossEntropyLoss()
            case "NLLLoss":
                self.__loss_fun = nn.NLLLoss()
            case str():
                raise RuntimeError(f"unknown loss function {loss_fun}")
            case _:
                self.__loss_fun = loss_fun

    def offload_from_memory(self):
        self.model.zero_grad(set_to_none=True)
        self.to(device="cpu")

    def get_input_feature(self, inputs):
        if hasattr(self.model, "get_input_feature"):
            return self.model.get_input_feature(inputs)
        return None

    def split_batch_input(self, inputs, targets) -> tuple:
        batch_dim = 0
        return inputs, batch_dim

    def __call__(
        self,
        inputs: Any,
        targets: Any,
        phase: MachineLearningPhase | None = None,
        device: None | torch.device = None,
        non_blocking: bool = False,
        is_input_feature: bool = False,
        need_backward: bool = False,
        **kwargs: Any,
    ) -> dict:
        if phase is not None:
            self.__set_model_mode(
                is_training=(phase == MachineLearningPhase.Training),
                need_backward=need_backward,
            )

        # deal with nested targets
        if self.model_type == ModelType.Classification and isinstance(
            targets, torch.Tensor
        ):
            targets = targets.view(-1)

        if device is not None:
            inputs = tensor_to(inputs, device=device, non_blocking=non_blocking)
            targets = tensor_to(targets, device=device, non_blocking=non_blocking)
            self.to(device=device, non_blocking=non_blocking)

        input_features = inputs
        if not is_input_feature and self.need_input_features:
            input_features = self.get_input_feature(inputs)
        return self._forward_model(
            inputs=inputs,
            is_input_feature=is_input_feature,
            targets=targets,
            non_blocking=non_blocking,
            **kwargs,
        ) | {
            "inputs": inputs,
            "input_features": input_features,
            "targets": targets,
        }

    def _forward_model(
        self, inputs: Any, is_input_feature: bool, **kwargs: Any
    ) -> dict:
        fun: Callable = self.model
        if hasattr(self.model, "forward_input_feature") and is_input_feature:
            fun = self.model.forward_input_feature
        match inputs:
            case torch.Tensor():
                output = fun(inputs)
            case tuple():
                output = fun(*inputs)
            case dict():
                output = fun(**inputs)
            case _:
                raise NotImplementedError(type(inputs))
        return self._compute_loss(output=output, **kwargs)

    def _compute_loss(
        self, output: Any, targets: Any, non_blocking: bool, **kwargs: Any
    ) -> dict:
        match output:
            case torch.Tensor():
                match self.loss_fun:
                    case nn.BCEWithLogitsLoss():
                        output = output.view(-1)
                        targets = targets.to(
                            dtype=output.dtype, non_blocking=non_blocking
                        )
                loss = self.loss_fun(output, targets)
                return {
                    "loss": loss,
                    "model_output": output,
                    "is_averaged_loss": self.__is_averaged_loss(),
                }
        raise NotImplementedError()

    def replace_model(self, model):
        return ModelEvaluator(
            model=model,
            loss_fun=self.loss_fun,
            model_type=self.model_type,
        )

    def to(self, device, non_blocking=False):
        for param in self.model.parameters():
            if param.device != device:
                self.model.to(device=device, non_blocking=non_blocking)
            break
        for buffer in self.model.buffers():
            if buffer.device != device:
                self.model.to(device=device, non_blocking=non_blocking)
            return

    def __choose_loss_function(self) -> Callable:
        layers = [
            m
            for _, m in self.model_util.get_modules()
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

    def get_underlying_model(self) -> torch.nn.Module:
        match self.model:
            case torch.quantization.stubs.QuantWrapper():
                return self.model.module
            case _:
                return self.model

    def __is_averaged_loss(self) -> bool | None:
        if hasattr(self.loss_fun, "reduction"):
            if self.loss_fun.reduction in ("mean",):
                return True
            return False
        return None

    def __repr__(self):
        return f"model: {self._model.__class__.__name__}, loss_fun: {self.loss_fun}"

    def __set_model_mode(self, is_training: bool, need_backward: bool = False) -> None:
        if is_training:
            if self._model.training:
                return
            self._model.train()
            return
        if self._model.training:
            self._model.eval()
            if need_backward:
                self.model_util.change_modules(
                    f=lambda _, module, __: module.train(), module_type=nn.RNNBase
                )


# class CheckPointedModelWithLoss:
#     def __init__(self, model_evaluator:ModelEvaluator):
#         self.__model_evaluator = model_evaluator.replace_model(
#             get_checkpointed_model(model_evaluator.model)
#         )

#     def __getattr__(self, attr):
#         return getattr(self.__model_evaluator, attr)

#     def __call__(self, **kwargs) -> dict:
#         phase = kwargs["phase"]
#         if phase == MachineLearningPhase.Training:
#             input_features = kwargs.get("input_features", None)
#             if input_features is not None:
#                 input_features.requires_grad_()
#             inputs = kwargs.get("inputs", None)
#             if inputs is not None:
#                 inputs.requires_grad_()
#         return self.__model_evaluator.__call__(**kwargs)


class VisionModelEvaluator(ModelEvaluator):
    pass
    # def __call__(self, inputs, **kwargs):
    #     inputs = tensor_to(inputs, non_blocking=True, memory_format=torch.channels_last)
    #     return super().__call__(inputs=inputs, **kwargs)

    # def to(self, device, non_blocking=False):
    #     self.model.to(non_blocking=non_blocking, memory_format=torch.channels_last)
    #     super().to(device=device, non_blocking=non_blocking)


# class ParallelModelWithLoss(ModelEvaluator):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         assert torch.cuda.is_available()
#         if not torch.distributed.is_initialized():
#             torch.distributed.init_process_group(
#                 backend="nccl",
#                 init_method="tcp://127.0.0.1:23456",
#                 rank=0,
#                 world_size=len(get_devices()),
#             )
#         self._original_model = self._model
#         self._model = DDP(self._original_model)

#     @classmethod
#     def create(cls, model_evaluator:ModelEvaluator):
#         return cls(
#             model=model_evaluator.model,
#             loss_fun=model_evaluator.loss_fun,
#             model_type=model_evaluator.model_type,
#         )
