from collections.abc import Callable, Generator, Iterable
from typing import Any

import torch
import torch.nn
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order
from cyy_naive_lib.log import log_debug

from ..ml_type import BlockType, ModelGradient, ModelParameter, TensorDict
from ..tensor import cat_tensors_to_vector


class ModelUtil:
    def __init__(self, model: torch.nn.Module) -> None:
        self.__model: torch.nn.Module = model
        self.__previous_device: None | torch.device = None

    @property
    def model(self) -> torch.nn.Module:
        return self.__model

    def to_device(self, device: torch.device, non_blocking: bool = True) -> None:
        flag = self.__previous_device == device
        if flag:
            for param in self.model.parameters():
                if param.device != device:
                    flag = False
                    break
        if flag:
            for buffer in self.model.buffers():
                if buffer.device != device:
                    flag = False
                    break
        if not flag:
            self.model.to(device=device, non_blocking=non_blocking)
        self.__previous_device = device

    def get_parameter_seq(self, **kwargs: Any) -> Generator:
        return get_mapping_values_by_key_order(self.get_parameters(**kwargs))

    def get_parameter_list(self, **kwargs: Any) -> torch.Tensor:
        return cat_tensors_to_vector(self.get_parameter_seq(**kwargs))

    def load_parameters(
        self,
        parameters: ModelParameter,
        check_parameter: bool = False,
        keep_grad: bool = False,
    ) -> None:
        assert parameters
        for name, parameter in parameters.items():
            old_parameter: torch.Tensor | None = None
            if check_parameter or parameter.dtype == torch.float64:
                old_parameter = self.model.get_parameter(name)
            if old_parameter is not None:
                parameter = parameter.to(dtype=old_parameter.dtype)
            self.set_attr(name, parameter, as_parameter=True, keep_grad=keep_grad)

    def get_buffers(self) -> TensorDict:
        return dict(self.model.named_buffers())

    def load_buffers(self, buffers: TensorDict) -> None:
        for name, parameter in buffers.items():
            self.set_attr(name, parameter, as_parameter=False)

    def get_parameters(self, detach: bool = True) -> ModelParameter:
        return {
            name: parameter.detach() if detach else parameter
            for name, parameter in self.model.named_parameters()
        }

    def get_gradients(self) -> ModelGradient:
        return {
            k: v.grad
            for k, v in self.get_parameters(detach=False).items()
            if v.grad is not None
        }

    def load_gradients(self, gradients: ModelGradient) -> None:
        assert gradients
        for name, grad in gradients.items():
            self.set_grad(name, grad)

    def disable_running_stats(self) -> None:
        for _, module in self.get_modules():
            if hasattr(module, "track_running_stats"):
                module.track_running_stats = False
                module.register_buffer("running_mean", None)
                module.register_buffer("running_var", None)
                module.register_buffer("num_batches_tracked", None)

    def reset_running_stats(self) -> None:
        for _, module in self.get_modules():
            if hasattr(module, "reset_running_stats"):
                module.reset_running_stats()

    def set_grad(
        self,
        name: str,
        grad: torch.Tensor,
    ) -> None:
        module = self.model.get_submodule(name)
        module.grad = grad

    def set_attr(
        self,
        name: str,
        value: Any,
        as_parameter: bool = True,
        keep_grad: bool = False,
    ) -> None:
        module: torch.nn.Module = self.model
        components = name.split(".")

        if len(components) >= 2:
            module = self.model.get_submodule(".".join(components[0:-1]))
            component = components[-1]
        else:
            component = name
        if hasattr(module, component):
            delattr(module, component)
        if as_parameter:
            if keep_grad:
                value._is_param = True
                module._parameters[component] = value
            else:
                module.register_parameter(component, torch.nn.Parameter(value))
        else:
            module.register_buffer(component, value)

    def filter_modules(
        self,
        module_type: type | None = None,
        module_names: Iterable[str] | None = None,
    ) -> Generator:
        if module_names is not None:
            module_names = set(module_names)
        for name, module in self.get_modules():
            if (module_type is not None and isinstance(module, module_type)) or (
                module_names is not None and name in module_names
            ):
                yield name, module

    def change_modules(self, f: Callable, **kwargs) -> bool:
        flag = False
        for name, module in self.filter_modules(**kwargs):
            f(name, module, self)
            flag = True
        return flag

    def freeze_modules(self, **kwargs: Any) -> bool:
        def freeze(name, module, model_util) -> None:
            log_debug("freeze %s", name)
            module.fronzen_parameters = set()
            parameter_dict: ModelParameter = {}
            for param_name, parameter in module.named_parameters():
                parameter_dict[name + "." + param_name] = parameter.data
                module.fronzen_parameters.add(param_name)

            for k, v in parameter_dict.items():
                model_util.set_attr(k, v, as_parameter=False)

        return self.change_modules(f=freeze, **kwargs)

    def unfreeze_modules(self, **kwargs: Any) -> bool:
        def unfreeze(name, module, model_util) -> None:
            parameter_dict: ModelParameter = {}
            if not hasattr(module, "fronzen_parameters"):
                log_debug("nothing to unfreeze")
                return
            assert module.fronzen_parameters
            parameter_dict = {
                f"{name}.{param_name}": getattr(module, param_name)
                for param_name in module.fronzen_parameters
            }
            delattr(module, "fronzen_parameters")
            assert parameter_dict
            log_debug("unfreeze %s %s", name, parameter_dict.keys())
            for k, v in parameter_dict.items():
                model_util.set_attr(k, v, as_parameter=True)

        return self.change_modules(f=unfreeze, **kwargs)

    def have_module(self, module_type: type) -> bool:
        for _ in self.filter_modules(module_type=module_type):
            return True
        return False

    def get_modules(self) -> Generator:
        def get_module_impl(model: torch.nn.Module, prefix: str) -> Generator:
            yield prefix, model
            for name, module in model.named_children():
                if module is None:
                    continue
                module_prefix: str = prefix + ("." if prefix else "") + name
                if isinstance(module, torch.nn.Conv2d):
                    yield module_prefix, module
                    continue
                has_submodule = False
                for sub_name, sub_module in get_module_impl(module, module_prefix):
                    has_submodule = True
                    yield sub_name, sub_module
                if not has_submodule:
                    yield module_prefix, module

        return get_module_impl(self.model, "")

    def get_last_underlying_module(self) -> torch.nn.Module:
        module_pairs = list(
            reversed(
                [
                    (name, module)
                    for name, module in self.get_modules()
                    if not isinstance(
                        module,
                        torch.quantization.QuantStub
                        | torch.quantization.DeQuantStub
                        | torch.quantization.QuantWrapper
                        | torch.quantization.FakeQuantize
                        | torch.quantization.MovingAverageMinMaxObserver
                        | torch.quantization.MovingAveragePerChannelMinMaxObserver
                        | torch.nn.modules.dropout.Dropout,
                    )
                ]
            )
        )
        for name, module in module_pairs:
            if any(name2.startswith(f"{name}.") for name2, _ in module_pairs):
                continue
            return module
        raise RuntimeError()

    def get_module_blocks(
        self,
        block_types: set,
    ) -> list[BlockType]:
        def module_has_type(module, module_type) -> bool:
            match module_type:
                case str():
                    module_class_name = module.__class__.__name__
                    return (
                        module_class_name == module_type
                        or module_class_name.endswith("." + module_class_name)
                    )
                case _:
                    return isinstance(module, module_type)

        blocks: list[BlockType] = []
        modules = list(self.get_modules())
        while modules:
            end_index = None
            candidates: set = block_types
            for i, (module_name, module) in enumerate(modules):
                new_candidates = set()
                for candidate in candidates:
                    if module_has_type(module, candidate[0]) and (
                        i == 0
                        or len(module_name.split("."))
                        == len(modules[i - 1][0].split("."))
                    ):
                        if len(candidate) == 1:
                            end_index = i
                        else:
                            new_candidates.add(candidate[1:])
                if not new_candidates:
                    break
                candidates = new_candidates
            if end_index is not None:
                block_modules = modules[: end_index + 1]
                blocks.append(block_modules)
                modules = modules[end_index + 1 :]
                modules = [
                    p
                    for p in modules
                    if not any(p[0].startswith(q[0] + ".") for q in block_modules)
                ]
            else:
                modules = modules[1:]
        return blocks
