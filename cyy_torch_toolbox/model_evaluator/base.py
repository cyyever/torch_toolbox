import functools
from typing import Callable

import torch
from cyy_naive_lib.log import get_logger
# from cyy_torch_toolbox.device import get_devices
from cyy_torch_toolbox.ml_type import MachineLearningPhase, ModelType
# from cyy_torch_toolbox.model_transform.checkpointed_model import \
#     get_checkpointed_model
from cyy_torch_toolbox.model_util import ModelUtil
from cyy_torch_toolbox.tensor import tensor_to
from torch import nn

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
        self._original_model: torch.nn.Module = model
        self._model: torch.nn.Module = model
        self.__name = model_name
        self.__loss_fun: Callable | None = None
        if loss_fun is not None:
            self.set_loss_fun(loss_fun)
        self.__model_type = model_type
        self.need_input_features = False
        self.need_cpu_inputs = False

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
        if isinstance(loss_fun, str):
            if loss_fun == "CrossEntropyLoss":
                self.__loss_fun = nn.CrossEntropyLoss()
            else:
                raise RuntimeError(f"unknown loss function {loss_fun}")
        else:
            self.__loss_fun = loss_fun

    def offload_from_memory(self):
        self.model.zero_grad(set_to_none=True)
        self.to(device="cpu")

    def get_input_feature(self, inputs):
        if hasattr(self.model, "get_input_feature"):
            return self.model.get_input_feature(inputs)
        return None

    def __call__(
        self,
        inputs,
        targets,
        phase: MachineLearningPhase | None = None,
        device=None,
        non_blocking: bool = False,
        input_features=None,
        need_backward: bool = False,
        **kwargs,
    ) -> dict:
        if phase is not None:
            self.__set_model_mode(
                is_training=(phase == MachineLearningPhase.Training),
                need_backward=need_backward,
            )

        # DALI returns nested targets
        if len(targets.shape) > 1:
            targets = targets.view(-1).long()

        cpu_inputs = None
        if device is not None:
            if self.need_cpu_inputs:
                cpu_inputs = tensor_to(inputs, device="cpu", non_blocking=non_blocking)
            if input_features is not None:
                input_features = tensor_to(
                    input_features, device=device, non_blocking=non_blocking
                )
            else:
                inputs = tensor_to(inputs, device=device, non_blocking=non_blocking)
            targets = tensor_to(targets, device=device, non_blocking=non_blocking)
            self.to(device=device, non_blocking=non_blocking)

        if input_features is None and self.need_input_features:
            input_features = self.get_input_feature(inputs)
            assert input_features is not None
        output = self._forward_model(
            inputs=inputs, input_features=input_features, targets=targets, **kwargs
        )

        match output:
            case torch.Tensor():
                match self.loss_fun:
                    case nn.BCEWithLogitsLoss():
                        output = output.view(-1)
                        targets = targets.to(
                            dtype=output.dtype, non_blocking=non_blocking
                        )
                loss = self.loss_fun(output, targets)
                output = {"loss": loss, "classification_output": output}
        is_averaged_loss = self.__is_averaged_loss()
        if is_averaged_loss is None:
            is_averaged_loss = output["classification_output"] is not None
        return output | {
            "inputs": inputs,
            "cpu_inputs": cpu_inputs,
            "input_features": input_features,
            "targets": targets,
            "is_averaged_loss": is_averaged_loss,
        }

    def _forward_model(
        self, inputs, input_features=None, **kwargs
    ) -> dict | torch.Tensor:
        fun: Callable = self.model
        if hasattr(self.model, "forward_input_feature") and input_features is not None:
            fun = self.model.forward_input_feature
            real_inputs = input_features
        else:
            real_inputs = inputs
        match real_inputs:
            case torch.Tensor():
                return fun(real_inputs)
            case tuple():
                return fun(*real_inputs)
            case dict():
                return fun(**real_inputs)
            case _:
                raise NotImplementedError(type(real_inputs))

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

    def __choose_loss_function(self) -> torch.nn.modules.loss._Loss:
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

    def get_underlying_model(self):
        model = self._original_model
        if isinstance(model, torch.quantization.stubs.QuantWrapper):
            return model.module
        return model

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
