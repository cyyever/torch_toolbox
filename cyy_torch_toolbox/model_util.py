from typing import Any, Callable, Optional, Type

import torch
import torch.nn as nn
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order
from cyy_naive_lib.log import get_logger

from tensor import cat_tensors_to_vector, load_tensor_dict


class ModelUtil:
    def __init__(self, model: torch.nn.Module):
        self.__model = model
        self.__parameter_dict: Optional[dict] = None
        self.__parameter_shapes: Optional[dict] = None

    @property
    def model(self):
        return self.__model

    def get_parameter_seq(self, detach=True):
        res = self.get_parameter_dict(detach=detach)
        return get_mapping_values_by_key_order(res)

    def get_parameter_list(self, detach: bool = True) -> torch.Tensor:
        return cat_tensors_to_vector(self.get_parameter_seq(detach=detach))

    def load_parameter_list(
        self,
        parameter_list: torch.Tensor,
        check_parameter: bool = True,
        as_parameter: bool = True,
    ) -> None:
        if isinstance(parameter_list, torch.Tensor):
            parameter_shapes = self.get_parameter_shapes()
            assert parameter_shapes
            parameter_dict = load_tensor_dict(parameter_shapes, parameter_list)
        else:
            parameter_dict = self.get_parameter_dict()
            assert parameter_dict
            assert len(parameter_list) == len(parameter_dict)
            for name in sorted(parameter_dict.keys()):
                parameter_dict[name] = parameter_list.pop(0)
        self.load_parameter_dict(
            parameter_dict, check_parameter=check_parameter, as_parameter=as_parameter
        )

    def load_parameter_dict(
        self,
        parameter_dict: dict,
        check_parameter: bool = True,
        as_parameter: bool = True,
    ) -> None:
        assert parameter_dict
        for name, parameter in parameter_dict.items():
            if check_parameter:
                assert self.has_attr(name)
            self.set_attr(name, parameter, as_parameter=as_parameter)
        self.__parameter_dict = None

    def get_parameter_dict(self, detach: bool = True) -> dict:
        if self.__parameter_dict is not None:
            return self.__parameter_dict
        res: dict = {}
        for name, parameter in self.model.named_parameters():
            if detach:
                parameter = parameter.detach()
            res[name] = parameter
        self.__parameter_dict = res
        return self.__parameter_dict

    def get_parameter_shapes(self) -> dict:
        if self.__parameter_shapes is not None:
            return self.__parameter_shapes
        res: dict = {}
        for name, parameter in self.model.named_parameters():
            res[name] = parameter.shape
        self.__parameter_shapes = res
        return self.__parameter_shapes

    def get_gradient_list(self):
        return cat_tensors_to_vector(
            (
                parameter.grad
                for parameter in self.get_parameter_seq(detach=False)
                if parameter.grad is not None
            )
        )

    def disable_running_stats(self) -> None:
        def impl(_, module, __):
            # Those lines are copied from the official code
            module.track_running_stats = False
            module.register_buffer("running_mean", None)
            module.register_buffer("running_var", None)
            module.register_buffer("num_batches_tracked", None)

        self.change_modules(f=impl, module_type=torch.nn.modules.batchnorm._NormBase)

    def reset_running_stats(self) -> None:
        def impl(_, module, __):
            module.reset_running_stats()

        self.change_modules(f=impl, module_type=torch.nn.modules.batchnorm._NormBase)

    def register_module(self, name: str, module) -> None:
        if "." not in name:
            self.model.register_module(name, module)
        else:
            components = name.split(".")
            module = self.get_attr(".".join(components[:-1]))
            module.register_module(components[-1], module)

    def set_attr(self, name: str, value: Any, as_parameter: bool = True) -> None:
        model = self.model
        components = name.split(".")
        for i, component in enumerate(components):
            if i + 1 != len(components):
                model = getattr(model, component)
            else:
                if hasattr(model, component):
                    delattr(model, component)
                if as_parameter:
                    model.register_parameter(component, nn.Parameter(value))
                else:
                    model.register_buffer(component, value)
        if self.__parameter_dict is not None:
            self.__parameter_dict = None

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

    def change_modules(
        self,
        f: Callable,
        module_type: Type | None = None,
        module_name: str | None = None,
    ) -> None:
        has_module = False
        for name, module in self.get_modules():
            if (module_type is not None and isinstance(module, module_type)) or (
                module_name is not None and name == module_name
            ):
                f(name, module, self)
                has_module = True
        assert has_module

    def freeze_modules(self, **kwargs) -> None:
        def freeze(name, module, model_util):
            get_logger().info("freeze %s", name)
            parameter_dict = {}
            for param_name, parameter in module.named_parameters():
                parameter_dict[name + "." + param_name] = parameter.data
            for k, v in parameter_dict.items():
                model_util.set_attr(k, v, as_parameter=False)

        self.change_modules(f=freeze, **kwargs)

    def unfreeze_modules(self, **kwargs) -> None:
        def unfreeze(name, module, model_util):
            get_logger().info("unfreeze %s", name)
            parameter_dict = {}
            for param_name, parameter in module.named_parameters():
                parameter_dict[name + "." + param_name] = parameter.data
            for k, v in parameter_dict.items():
                model_util.set_attr(k, v, as_parameter=True)

        self.change_modules(f=unfreeze, **kwargs)

    def have_module(
        self, module_type: Type | None = None, module_name: str | None = None
    ) -> bool:
        for name, module in self.get_modules():
            if (module_type is not None and isinstance(module, module_type)) or (
                module_name is not None and name == module_name
            ):
                return True
        return False

    def get_modules(self) -> list[tuple[str, Any]]:
        def get_module_impl(model, prefix: str):
            result = [(prefix, model)]
            for name, module in model._modules.items():
                if module is None:
                    continue
                module_prefix: str = prefix + ("." if prefix else "") + name
                if isinstance(module, nn.Conv2d):
                    result.append((module_prefix, module))
                    continue
                sub_result = get_module_impl(module, module_prefix)
                if sub_result:
                    result += sub_result
                else:
                    result.append((module_prefix, module))
            return result

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
