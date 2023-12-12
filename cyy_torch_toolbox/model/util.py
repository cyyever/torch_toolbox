from typing import Any, Callable, Generator, Type

import torch
import torch.nn
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order
from cyy_naive_lib.log import get_logger

from ..tensor import cat_tensors_to_vector


class ModelUtil:
    def __init__(self, model: torch.nn.Module) -> None:
        self.__model: torch.nn.Module = model

    @property
    def model(self) -> torch.nn.Module:
        return self.__model

    def get_parameter_seq(self, **kwargs: Any) -> Generator:
        return get_mapping_values_by_key_order(self.get_parameter_dict(**kwargs))

    def get_parameter_list(self, **kwargs: Any) -> torch.Tensor:
        return cat_tensors_to_vector(self.get_parameter_seq(**kwargs))

    def load_parameter_dict(
        self,
        parameter_dict: dict,
        check_parameter: bool = False,
        keep_grad: bool = False,
    ) -> None:
        assert parameter_dict
        for name, parameter in parameter_dict.items():
            if check_parameter:
                assert self.has_attr(name)
            self.set_attr(name, parameter, as_parameter=True, keep_grad=keep_grad)

    def get_buffer_dict(self) -> dict:
        return dict(self.model.named_buffers())

    def load_buffer_dict(self, buffer_dict: dict) -> None:
        for name, parameter in buffer_dict.items():
            self.set_attr(name, parameter, as_parameter=False)

    def clear_parameters(self) -> None:
        def clear(module: torch.nn.Module) -> None:
            module._parameters = {k: None for k in module._parameters}

        for _, module in self.get_modules():
            clear(module)

    def get_parameter_dict(self, detach: bool = True) -> dict:
        res: dict = {}
        for name, parameter in self.model.named_parameters():
            if detach:
                parameter = parameter.detach()
            res[name] = parameter
        return res

    def get_gradient_dict(self) -> dict:
        return {
            k: v.grad
            for k, v in self.get_parameter_dict(detach=False).items()
            if v.grad is not None
        }

    def disable_running_stats(self) -> None:
        def impl(_, module, __) -> None:
            module.track_running_stats = False
            module.register_buffer("running_mean", None)
            module.register_buffer("running_var", None)
            module.register_buffer("num_batches_tracked", None)

        self.change_modules(f=impl, module_type=torch.nn.modules.batchnorm._NormBase)

    def reset_running_stats(self) -> None:
        for _, module in self.get_modules():
            if hasattr(module, "reset_running_stats"):
                module.reset_running_stats()

    def register_module(self, name: str, module: Any) -> None:
        if "." not in name:
            self.model.register_module(name, module)
        else:
            components = name.split(".")
            module = self.get_attr(".".join(components[:-1]))
            module.register_module(components[-1], module)

    def set_attr(
        self, name: str, value: Any, as_parameter: bool = True, keep_grad: bool = False
    ) -> None:
        model = self.model
        components = name.split(".")
        for i, component in enumerate(components):
            if i + 1 != len(components):
                model = getattr(model, component)
            else:
                if hasattr(model, component):
                    delattr(model, component)
                if as_parameter:
                    if keep_grad:
                        setattr(value, "_is_param", True)
                        model._parameters[component] = value
                    else:
                        model.register_parameter(component, torch.nn.Parameter(value))
                else:
                    model.register_buffer(component, value)

    def get_attr(self, name: str) -> Any:
        val = self.model
        components = name.split(".")
        for component in components:
            val = getattr(val, component)
        return val

    def del_attr(self, name: str) -> None:
        model = self.model
        components = name.split(".")
        for i, component in enumerate(components):
            if i + 1 != len(components):
                model = getattr(model, component)
            else:
                delattr(model, component)

    def has_attr(self, name: str) -> bool:
        model = self.model
        components = name.split(".")
        for component in components:
            if not hasattr(model, component):
                return False
            model = getattr(model, component)
        return True

    def filter_modules(
        self,
        module_type: Type | None = None,
        module_name: str | None = None,
    ) -> Generator:
        for name, module in self.get_modules():
            if (module_type is not None and isinstance(module, module_type)) or (
                module_name is not None and name == module_name
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
            get_logger().info("freeze %s", name)
            parameter_dict = {}
            for param_name, parameter in module.named_parameters():
                parameter_dict[name + "." + param_name] = parameter.data
            for k, v in parameter_dict.items():
                model_util.set_attr(k, v, as_parameter=False)

        return self.change_modules(f=freeze, **kwargs)

    def unfreeze_modules(self, **kwargs: Any) -> bool:
        def unfreeze(name, module, model_util) -> None:
            get_logger().info("unfreeze %s", name)
            parameter_dict = {}
            for param_name, parameter in module.named_parameters():
                parameter_dict[name + "." + param_name] = parameter.data
            for k, v in parameter_dict.items():
                model_util.set_attr(k, v, as_parameter=True)

        return self.change_modules(f=unfreeze, **kwargs)

    def have_module(
        self, module_type: Type | None = None, module_name: str | None = None
    ) -> bool:
        for _ in self.filter_modules(module_type=module_type, module_name=module_name):
            return True
        return False

    def get_modules(self) -> Generator:
        def get_module_impl(model: torch.nn.Module, prefix: str) -> Generator:
            yield prefix, model
            for name, module in model._modules.items():
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

    def get_module_blocks(
        self,
        block_types: set,
    ) -> list:
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

        blocks: list = []
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
                modules = modules[end_index + 1:]
                modules = [
                    p
                    for p in modules
                    if not any(p[0].startswith(q[0] + ".") for q in block_modules)
                ]
            else:
                modules = modules[1:]
        return blocks
