import copy
from collections.abc import Callable, Iterable
from typing import Any

import torch
from cyy_naive_lib.log import log_debug, log_error
from torch import nn

from ..dataset import DatasetUtil
from ..ml_type import EvaluationMode, ModelType
from ..tensor import tensor_to
from .util import ModelUtil


class ModelEvaluator:
    """
    Combine a model with a loss function.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_type: None | ModelType = None,
        loss_fun: str | Callable | None = None,
        frozen_modules: dict | None = None,
        **kwargs,
    ) -> None:
        self._model: torch.nn.Module = model
        self._model_kwargs = copy.deepcopy(kwargs)
        self.__loss_fun: Callable | None = None
        self.__loss_fun_type: type | None = None
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
                assert self.model_util.freeze_modules(module_names=names)
            case None:
                pass
            case _:
                raise NotImplementedError(frozen_modules)
        self.__evaluation_kwargs: dict = {}

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def model_util(self) -> ModelUtil:
        return ModelUtil(self.model)

    @property
    def model_type(self) -> ModelType:
        return self.__model_type

    @property
    def loss_fun(self) -> Callable:
        if self.__loss_fun is None:
            if self.__loss_fun_type is None:
                self.__loss_fun_type = self._choose_loss_function_type()
            if self.__loss_fun_type is not None:
                self.__loss_fun = self.__loss_fun_type()
            else:
                self.__loss_fun = self._choose_loss_function()
        assert self.__loss_fun is not None
        return self.__loss_fun

    def set_model(self, model) -> None:
        self._model = model

    def set_forward_fun(self, forward_fun: str) -> None:
        self.add_evaluation_kwargs(forward_fun=forward_fun)

    def add_evaluation_kwargs(self, **kwargs) -> None:
        self.__evaluation_kwargs.update(kwargs)

    def remove_evaluation_kwargs(self, keys: Iterable[str]) -> None:
        for key in keys:
            self.__evaluation_kwargs.pop(key, None)

    def set_loss_fun(self, loss_fun: Callable | str) -> None:
        match loss_fun:
            case "CrossEntropyLoss":
                self.__loss_fun_type = nn.CrossEntropyLoss
            case "NLLLoss":
                self.__loss_fun_type = nn.NLLLoss
            case str():
                raise RuntimeError(f"unknown loss function {loss_fun}")
            case _:
                self.__loss_fun = loss_fun
                self.__loss_fun_type = type(loss_fun)
                return
        assert self.__loss_fun_type is not None
        self.__loss_fun = self.__loss_fun_type()

    def offload_from_device(self) -> None:
        self.model_util.to_device(device=torch.device("cpu"))

    def get_input_feature(self, inputs: Any) -> Any:
        fun = getattr(self.model, "get_input_feature", None)
        if fun is not None:
            return fun(inputs)
        return None

    def split_batch_input(self, inputs: Any, **kwargs: Any) -> dict:
        return {"inputs": inputs, "batch_dim": 0}

    def backward(
        self,
        loss: torch.Tensor,
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
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        **backward_kwargs,
    ) -> None:
        self.backward(loss=loss, optimizer=optimizer, **backward_kwargs)
        optimizer.step()

    def get_underlying_model(self) -> torch.nn.Module:
        match self.model:
            case torch.quantization.QuantWrapper():
                return self.model.module
            case _:
                return self.model

    def get_normalized_batch_loss(
        self, dataset_util: DatasetUtil, forward_result: dict
    ) -> Any:
        if forward_result["is_averaged_loss"]:
            sample_number = 0
            if self.model_type == ModelType.TokenClassification:
                sample_number = dataset_util.sample_number
            else:
                sample_number = len(dataset_util)
            return (
                forward_result["loss"]
                * forward_result["loss_batch_size"]
                / sample_number
            )
        return None

    def __call__(
        self,
        *,
        device: None | torch.device = None,
        inputs: Any | None = None,
        evaluation_mode: EvaluationMode | None = None,
        **kwargs: Any,
    ) -> dict:
        if evaluation_mode is not None:
            self.__set_model_mode(evaluation_mode=evaluation_mode)
        raw_inputs = inputs
        non_blocking = kwargs.get("non_blocking", True)
        if device is not None:
            if inputs is not None:
                inputs = tensor_to(inputs, device=device, non_blocking=non_blocking)
            tensor_to(kwargs, non_blocking=non_blocking)
            self.model_util.to_device(device=device)
        return {
            "inputs": inputs,
            "targets": kwargs.get("targets"),
            "raw_inputs": raw_inputs,
        } | self._forward_model(
            inputs=inputs,
            device=device,
            evaluation_mode=evaluation_mode,
            **(kwargs | self.__evaluation_kwargs),
        )

    def _get_forward_fun(self) -> Callable:
        fun: Callable = self.model
        if "forward_fun" in self.__evaluation_kwargs:
            fun = self.__evaluation_kwargs["forward_fun"]
            if isinstance(fun, str):
                fun = getattr(self.model, fun)
            log_debug("forward with function %s", fun)
        return fun

    def _forward_model(self, *, inputs: Any, **kwargs: Any) -> dict:
        fun: Callable = self._get_forward_fun()
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
        res = {
            "original_output": output,
        }

        convert_kwargs = {"device": output.device}
        assert isinstance(output, torch.Tensor)
        loss_fun = self.loss_fun
        if not reduce_loss:
            assert self.__loss_fun_type is not None
            loss_fun = self.__loss_fun_type(reduction="none")

        if len(output.shape) == 2 and output.shape[-1] == 1:
            output = output.view(-1)
        match loss_fun:
            case nn.BCEWithLogitsLoss():
                if targets.dtype is torch.long or targets.dtype is torch.int:
                    convert_kwargs["dtype"] = torch.float
        targets = targets.to(**convert_kwargs, non_blocking=True)
        loss = loss_fun(output, targets)
        res |= {
            "loss": loss,
            "model_output": output,
            "is_averaged_loss": self.__is_averaged_loss(loss_fun),
        }
        if kwargs["evaluation_mode"] != EvaluationMode.SampleInference:
            res |= {"loss_batch_size": (targets.view(-1) != -100).sum()}
        return res

    def _choose_loss_function(self) -> Callable:
        raise NotImplementedError()

    def _choose_loss_function_type(self) -> type | None:
        last_module = self.model_util.get_last_underlying_module()

        log_debug("last module is %s", last_module.__class__)
        loss_fun_type: None | type = None
        match last_module:
            case nn.LogSoftmax():
                log_debug("choose loss function NLLLoss")
                loss_fun_type = nn.NLLLoss
            case nn.Linear():
                if last_module.out_features > 1:
                    log_debug("choose loss function CrossEntropyLoss")
                    loss_fun_type = nn.CrossEntropyLoss
                else:
                    log_debug("choose loss function BCEWithLogitsLoss")
                    loss_fun_type = nn.BCEWithLogitsLoss
        if loss_fun_type is None:
            log_error("can't choose a loss function, model is %s", self._model)
            raise NotImplementedError(type(last_module))
        return loss_fun_type

    @classmethod
    def __is_averaged_loss(cls, loss_fun) -> bool:
        if hasattr(loss_fun, "reduction"):
            match loss_fun.reduction:
                case "mean":
                    return True
        return False

    def __repr__(self) -> str:
        return f"model: {self._model.__class__.__name__}, loss_fun: {self.loss_fun}"

    def load_model_for_inference(self, model: torch.nn.Module) -> None:
        if self.model is model:
            return
        self.model.load_state_dict(model.state_dict())

    def __set_model_mode(self, evaluation_mode: EvaluationMode) -> None:
        modules: set[torch.nn.Module] = set()
        if hasattr(self._model, "__cyy_check_modules"):
            modules = getattr(self._model, "__cyy_check_modules")
        else:
            check_modules: set[torch.nn.Module] = {self._model}
            while check_modules:
                module = next(iter(check_modules))
                check_modules.remove(module)
                modules.add(module)
                for k in dir(module):
                    if "k".startswith("_"):
                        continue
                    if "loss" in k:
                        continue
                    v = getattr(module, k)
                    if isinstance(v, torch.nn.Module) and v not in modules:
                        # print("add sub-module ", k)
                        check_modules.add(v)
                        modules.add(v)
            setattr(self._model, "__cyy_check_modules", modules)

        for module in modules:
            match evaluation_mode:
                case EvaluationMode.Training:
                    if not module.training:
                        module.train()
                case EvaluationMode.Test | EvaluationMode.TestWithGrad:
                    if module.training:
                        module.eval()
                        if evaluation_mode == EvaluationMode.TestWithGrad:
                            ModelUtil(module).change_modules(
                                f=lambda _, sub_module, __: sub_module.train(),
                                module_type=nn.RNNBase,
                            )
