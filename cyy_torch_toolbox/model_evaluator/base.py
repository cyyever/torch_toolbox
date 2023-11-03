import functools
from typing import Any, Callable, Type

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
    ) -> None:
        self._model: torch.nn.Module = model
        self.__name = model_name
        self.__loss_fun: Callable | None = None
        self.__non_reduction_loss_fun: Callable | None = None
        if loss_fun is not None:
            self.set_loss_fun(loss_fun)
        if model_type is None:
            model_type = ModelType.Classification
        self.__model_type: ModelType = model_type

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
    def model_type(self) -> ModelType:
        return self.__model_type

    def get_underlying_model_type(self) -> ModelType:
        return self.model_type

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
        self.__non_reduction_loss_fun = None

    def offload_from_device(self) -> None:
        self.model.zero_grad(set_to_none=True)
        self.to(device=torch.device("cpu"))

    def get_input_feature(self, inputs: Any) -> None | Any:
        if hasattr(self.model, "get_input_feature"):
            return self.model.get_input_feature(inputs)
        return None

    def split_batch_input(self, inputs: Any, targets: Any) -> tuple:
        return inputs, 0

    def __call__(
        self,
        *,
        inputs: Any,
        targets: Any,
        phase: MachineLearningPhase | None = None,
        device: None | torch.device = None,
        non_blocking: bool = False,
        is_input_feature: bool = False,
        need_backward: bool = False,
        reduce_loss: bool = True,
        **kwargs: Any,
    ) -> dict:
        if need_backward:
            assert reduce_loss
        if phase is not None:
            self.__set_model_mode(
                is_training=(phase == MachineLearningPhase.Training),
                need_backward=need_backward,
            )

        if device is not None:
            inputs = tensor_to(inputs, device=device, non_blocking=non_blocking)
            targets = tensor_to(targets, device=device, non_blocking=non_blocking)
            self.to(device=device, non_blocking=non_blocking)

        return self._forward_model(
            inputs=inputs,
            is_input_feature=is_input_feature,
            targets=targets,
            non_blocking=non_blocking,
            device=device,
            reduce_loss=reduce_loss,
            **kwargs,
        ) | {"inputs": inputs, "targets": targets}

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
        self,
        *,
        output: Any,
        targets: Any,
        non_blocking: bool,
        reduce_loss: bool,
        **kwargs: Any,
    ) -> dict:
        original_output = output
        convert_kwargs = {"device": output.device}
        assert isinstance(output, torch.Tensor)
        loss_fun = self.loss_fun
        if not reduce_loss:
            if self.__non_reduction_loss_fun is None:
                self.__non_reduction_loss_fun = self.__choose_loss_function(
                    reduction=False
                )
            loss_fun = self.__non_reduction_loss_fun

        match loss_fun:
            case nn.CrossEntropyLoss():
                if len(targets.shape) > 1:
                    convert_kwargs["dtype"] = torch.float
                targets = targets.to(**convert_kwargs, non_blocking=non_blocking)
            case nn.BCEWithLogitsLoss():
                convert_kwargs["dtype"] = output.dtype
                targets = targets.to(**convert_kwargs, non_blocking=non_blocking).view(
                    -1
                )
                output = output.view(-1)
        loss = loss_fun(output, targets)
        res = {
            "loss": loss,
            "targets": targets,
            "original_output": original_output,
            "model_output": output,
            "is_averaged_loss": self.__is_averaged_loss(loss_fun),
        }
        if res["is_averaged_loss"]:
            res["loss_batch_size"] = targets.shape[0]
        return res

    def replace_model(self, model):
        return ModelEvaluator(
            model=model,
            loss_fun=self.loss_fun,
            model_type=self.model_type,
        )

    def to(self, device: torch.device, non_blocking: bool = False) -> None:
        self._to(model=self.model, device=device, non_blocking=non_blocking)

    def _to(
        self, model: torch.nn.Module, device: torch.device, non_blocking: bool
    ) -> None:
        for param in model.parameters():
            if param.device != device:
                model.to(device=device, non_blocking=non_blocking)
            break
        for buffer in model.buffers():
            if buffer.device != device:
                model.to(device=device, non_blocking=non_blocking)
            return

    def __choose_loss_function(self, reduction: bool = True) -> Callable:
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
        loss_fun_type: None | Type = None
        match last_layer:
            case nn.LogSoftmax():
                get_logger().debug("choose loss function NLLLoss")
                loss_fun_type = nn.NLLLoss
            case nn.Linear():
                if last_layer.out_features > 1:
                    get_logger().debug("choose loss function CrossEntropyLoss")
                    loss_fun_type = nn.CrossEntropyLoss
                else:
                    get_logger().debug("choose loss function BCEWithLogitsLoss")
                    loss_fun_type = nn.BCEWithLogitsLoss
        if loss_fun_type is None:
            get_logger().error("can't choose a loss function, model is %s", self._model)
            raise NotImplementedError(type(last_layer))
        if reduction:
            return loss_fun_type()
        return loss_fun_type(reduction="none")

    def get_underlying_model(self) -> torch.nn.Module:
        match self.model:
            case torch.quantization.stubs.QuantWrapper():
                return self.model.module
            case _:
                return self.model

    @classmethod
    def __is_averaged_loss(cls, loss_fun) -> bool:
        if hasattr(loss_fun, "reduction"):
            match loss_fun.reduction:
                case "mean":
                    return True
        return False

    def __repr__(self) -> str:
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
