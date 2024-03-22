import functools
from typing import Any, Callable, Iterable, Type

import torch
from cyy_naive_lib.log import get_logger
from torch import nn

from ..ml_type import EvaluationMode, ModelType
from ..tensor import tensor_to
from .util import ModelUtil

# from cyy_torch_toolbox.model_transform.checkpointed_model import \
#     get_checkpointed_model
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
        frozen_modules: dict | None = None,
        **kwargs,
    ) -> None:
        self._model: torch.nn.Module = model
        self.__name = model_name
        self.__loss_fun: Callable | None = None
        if loss_fun is not None:
            self.set_loss_fun(loss_fun)
        self.__model_type: ModelType = (
            model_type if model_type is not None else ModelType.Classification
        )
        assert "model_path" not in kwargs
        match frozen_modules:
            case {"types": types}:
                for t in types:
                    assert self.model_util.freeze_modules(module_type=t)
            case {"names": names}:
                for name in names:
                    assert self.model_util.freeze_modules(module_name=name)
            case None:
                pass
            case _:
                raise NotImplementedError(frozen_modules)
        self.__evaluation_kwargs: dict = {}

    def set_forward_fun(self, forward_fun: str) -> None:
        self.add_evaluation_kwargs(forward_fun=forward_fun)

    def add_evaluation_kwargs(self, **kwargs) -> None:
        self.__evaluation_kwargs.update(kwargs)

    def remove_evaluation_kwargs(self, keys: Iterable[str]) -> None:
        for key in keys:
            self.__evaluation_kwargs.pop(key, None)

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

    @property
    def loss_fun(self) -> Callable:
        if self.__loss_fun is None:
            self.__loss_fun = self._choose_loss_function()
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

    def offload_from_device(self) -> None:
        self.model_util.to_device(device=torch.device("cpu"))

    def get_input_feature(self, inputs: Any) -> Any:
        if hasattr(self.model, "get_input_feature"):
            return self.model.get_input_feature(inputs)
        return None

    def split_batch_input(self, inputs: Any, batch_size: int) -> dict:
        return {"inputs": inputs, "batch_dim": 0}

    def __call__(
        self,
        *,
        inputs: Any,
        targets: Any | None = None,
        device: None | torch.device = None,
        evaluation_mode: EvaluationMode | None = None,
        **kwargs: Any,
    ) -> dict:
        if evaluation_mode is not None:
            self.__set_model_mode(evaluation_mode=evaluation_mode)
        raw_inputs = inputs

        if device is not None:
            inputs = tensor_to(inputs, device=device, non_blocking=True)
            targets = tensor_to(targets, device=device, non_blocking=True)
            self.model_util.to_device(device=device)

        return {
            "inputs": inputs,
            "targets": targets,
            "raw_inputs": raw_inputs,
        } | self._forward_model(
            inputs=inputs,
            targets=targets,
            device=device,
            **(kwargs | self.__evaluation_kwargs),
        )

    def _forward_model(self, inputs: Any, **kwargs: Any) -> dict:
        fun: Callable = self.model
        if "forward_fun" in self.__evaluation_kwargs:
            forward_fun_name = self.__evaluation_kwargs["forward_fun"]
            fun = getattr(self.model, forward_fun_name)
            get_logger().debug("forward with function %s", forward_fun_name)
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
        reduce_loss: bool = True,
        **kwargs: Any,
    ) -> dict:
        original_output = output
        res = {
            "original_output": original_output,
        }
        if targets is None:
            return res
        convert_kwargs = {"device": output.device}
        assert isinstance(output, torch.Tensor)
        loss_fun = self.loss_fun
        if not reduce_loss:
            loss_fun = type(loss_fun)(reduction="none")

        match loss_fun:
            case nn.CrossEntropyLoss():
                if len(targets.shape) == 2 and targets.shape[-1] == 1:
                    targets = targets.view(-1)
                    res["targets"] = targets
                if len(targets.shape) > 1:
                    convert_kwargs["dtype"] = torch.float
                targets = targets.to(**convert_kwargs, non_blocking=True)
            case nn.BCEWithLogitsLoss():
                convert_kwargs["dtype"] = output.dtype
                targets = targets.to(**convert_kwargs, non_blocking=True).view(-1)
                output = output.view(-1)
        loss = loss_fun(output, targets)
        res |= {
            "loss": loss,
            "model_output": output,
            "is_averaged_loss": self.__is_averaged_loss(loss_fun),
            "loss_batch_size": targets.shape[0],
        }
        return res

    def backward(
        self,
        loss,
        optimizer: None | torch.optim.Optimizer = None,
        **backward_kwargs,
    ) -> None:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        else:
            self._model.zero_grad(set_to_none=True)
        loss.backward(**backward_kwargs)

    def backward_and_step(
        self,
        loss,
        optimizer: torch.optim.Optimizer,
        **backward_kwargs,
    ) -> None:
        self.backward(loss=loss, optimizer=optimizer, **backward_kwargs)
        optimizer.step()

    def get_normalized_batch_loss(self, dataset_size: int, forward_result: dict) -> Any:
        if forward_result["is_averaged_loss"]:
            assert dataset_size > 0
            return (
                forward_result["loss"]
                * forward_result["loss_batch_size"]
                / dataset_size
            )
        return None

    def set_model(self, model) -> None:
        self._model = model

    def _choose_loss_function(self) -> Callable:
        layers = [
            m
            for _, m in self.model_util.get_modules()
            if not isinstance(
                m,
                (
                    torch.quantization.QuantStub,
                    torch.quantization.DeQuantStub,
                    torch.quantization.QuantWrapper,
                    torch.quantization.FakeQuantize,
                    torch.quantization.MovingAverageMinMaxObserver,
                    torch.quantization.MovingAveragePerChannelMinMaxObserver,
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
        return loss_fun_type()

    def get_underlying_model(self) -> torch.nn.Module:
        match self.model:
            case torch.quantization.QuantWrapper():
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

    def __set_model_mode(self, evaluation_mode: EvaluationMode) -> None:
        match evaluation_mode:
            case EvaluationMode.Training:
                if not self._model.training:
                    self._model.train()
                    return
            case EvaluationMode.Test | EvaluationMode.TestWithGrad:
                if self._model.training:
                    self._model.eval()
                if evaluation_mode == EvaluationMode.TestWithGrad:
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
