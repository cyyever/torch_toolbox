from typing import Any, Callable, Generator, Type

import torch
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order
from cyy_naive_lib.log import get_logger

from .tensor import (cat_tensors_to_vector, load_tensor_dict,
                     load_tensor_dict_from_seq)


class ModelUtil:
    def __init__(self, model: torch.nn.Module) -> None:
        self.__model = model
        self.__parameter_shapes: dict | None = None
        self.__cached_buffer_names: set | None = None

    @property
    def model(self):
        return self.__model

    def get_parameter_seq(self, detach: bool = True) -> Generator:
        res = self.get_parameter_dict(detach=detach)
        return get_mapping_values_by_key_order(res)

    def get_parameter_list(self, detach: bool = True) -> torch.Tensor:
        return cat_tensors_to_vector(self.get_parameter_seq(detach=detach))

    def load_parameter_seq(
        self,
        parameter_seq: list,
        check_parameter: bool = False,
        as_parameter: bool = True,
        parameter_shapes: None | dict = None,
    ) -> None:
        if parameter_shapes is None:
            parameter_shapes = self.get_parameter_shapes()
        assert parameter_shapes
        parameter_dict = load_tensor_dict_from_seq(parameter_shapes, parameter_seq)
        self.load_parameter_dict(
            parameter_dict,
            check_parameter=check_parameter,
            as_parameter=as_parameter,
            update_parameter_shapes=False,
        )

    def load_parameter_list(
        self,
        parameter_list: torch.Tensor,
        check_parameter: bool = False,
        as_parameter: bool = True,
        parameter_shapes: None | dict = None,
    ) -> None:
        if parameter_shapes is None:
            parameter_shapes = self.get_parameter_shapes()
        assert parameter_shapes
        parameter_dict = load_tensor_dict(parameter_shapes, parameter_list)
        self.load_parameter_dict(
            parameter_dict,
            check_parameter=check_parameter,
            as_parameter=as_parameter,
            update_parameter_shapes=False,
        )

    def load_parameter_dict(
        self,
        parameter_dict: dict,
        check_parameter: bool = False,
        as_parameter: bool = True,
        update_parameter_shapes: bool = True,
    ) -> None:
        assert parameter_dict
        for name, parameter in parameter_dict.items():
            if check_parameter:
                assert self.has_attr(name)
            self.set_attr(name, parameter, as_parameter=as_parameter)
        if update_parameter_shapes:
            self.__parameter_shapes = None

    def get_buffer_dict(self) -> dict:
        return dict(self.model.named_buffers())

    def load_buffer_dict(self, buffer_dict: dict) -> None:
        for name, parameter in buffer_dict.items():
            self.set_attr(name, parameter, as_parameter=False)

    def clear_parameters(self) -> None:
        def clear(module) -> None:
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

    def get_parameter_shapes(self) -> dict:
        if self.__parameter_shapes is not None:
            return self.__parameter_shapes
        res: dict = {}
        for name, parameter in self.model.named_parameters():
            res[name] = parameter.shape
        self.__parameter_shapes = res
        return self.__parameter_shapes

    def get_gradient_list(self):
        try:
            return cat_tensors_to_vector(
                (parameter.grad for parameter in self.get_parameter_seq(detach=False))
            )
        except BaseException as e:
            for k, v in self.get_parameter_dict(detach=False).items():
                if v.grad is None:
                    raise NotImplementedError(k) from e
            raise e

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

    def change_modules(
        self,
        f: Callable,
        module_type: Type | None = None,
        module_name: str | None = None,
    ) -> None:
        for name, module in self.get_modules():
            if (module_type is not None and isinstance(module, module_type)) or (
                module_name is not None and name == module_name
            ):
                f(name, module, self)

    def cache_buffer_names(self) -> None:
        self.__cached_buffer_names = set()
        for param_name, _ in self.__model.named_buffers():
            self.__cached_buffer_names.add(param_name)

    @property
    def cached_buffer_names(self):
        return self.__cached_buffer_names

    def freeze_modules(self, **kwargs) -> None:
        def freeze(name, module, model_util) -> None:
            get_logger().info("freeze %s", name)
            parameter_dict = {}
            for param_name, parameter in module.named_parameters():
                parameter_dict[name + "." + param_name] = parameter.data
            for k, v in parameter_dict.items():
                model_util.set_attr(k, v, as_parameter=False)

        self.change_modules(f=freeze, **kwargs)

    def unfreeze_modules(self, **kwargs) -> None:
        def unfreeze(name, module, model_util) -> None:
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
        def get_module_impl(model: Any, prefix: str) -> list[tuple[str, Any]]:
            result = [(prefix, model)]
            for name, module in model._modules.items():
                if module is None:
                    continue
                module_prefix: str = prefix + ("." if prefix else "") + name
                if isinstance(module, torch.nn.Conv2d):
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
