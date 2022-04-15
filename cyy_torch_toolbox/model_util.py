import copy
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

    def remove_statistical_variables(self):
        for k in list(self.model.state_dict().keys()):
            if ".running_var" in k or ".running_mean" in k:
                get_logger().debug("remove %s from model", k)
                self.set_attr(k, None, as_parameter=False)
            elif k.startswith(".running_"):
                raise RuntimeError(f"unchecked key {k}")

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

    def change_sub_modules(self, sub_module_type: Type, f: Callable) -> None:
        for k, v in self.get_sub_modules():
            if isinstance(v, sub_module_type):
                f(k, v, self)

    def freeze_sub_modules(self, sub_module_type: Type) -> None:
        def freeze(name, sub_module, model_util):
            get_logger().info("freeze %s", name)
            parameter_dict = {}
            for param_name, parameter in sub_module.named_parameters():
                parameter_dict[name + "." + param_name] = parameter.data
            for k, v in parameter_dict.items():
                model_util.set_attr(k, v, as_parameter=False)

        self.change_sub_modules(sub_module_type, freeze)

    def has_sub_module(self, module_type: Type) -> bool:
        return any(
            isinstance(sub_module, module_type)
            for _, sub_module in self.get_sub_modules()
        )

    def get_sub_modules(self) -> list:
        def get_sub_module_impl(model, prefix):
            result = []
            for name, module in model._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                if isinstance(module, nn.Conv2d):
                    result.append((submodule_prefix, module))
                    continue
                sub_result = get_sub_module_impl(module, submodule_prefix)
                if sub_result:
                    result += sub_result
                else:
                    result.append((submodule_prefix, module))
            return result

        return get_sub_module_impl(self.model, "")

    def get_sub_module_blocks(
        self,
        block_types: set = None,
        block_classes: set = None,
        only_block_name: bool = True,
    ) -> list:
        if block_types is None:
            block_types = {
                (nn.Conv2d, nn.BatchNorm2d, nn.ReLU),
                (nn.BatchNorm2d, nn.ReLU, nn.Conv2d),
                (nn.BatchNorm2d, nn.Conv2d),
                (nn.Conv2d, nn.ReLU),
                (nn.Conv2d, nn.ReLU, nn.MaxPool2d),
                (nn.Linear, nn.ReLU),
            }
        if block_classes is None:
            block_classes = {"Bottleneck", "DenseBlock"}
        blocks: list = []
        i = 0
        memo: set = set()

        # use while loop to avoid add submodule when a module is added
        while True:
            flag = False
            for module in self.model.named_modules(memo=copy.copy(memo)):
                module_class_name = module[1].__class__.__name__
                if any(
                    (
                        module_class_name == block_class
                        or module_class_name.endswith("." + block_class)
                        for block_class in block_classes
                    )
                ):
                    blocks.append([module])
                    memo.add(module[1])
                    flag = True
                    break
            if not flag:
                break

        modules = list(self.model.named_modules(memo=copy.copy(memo)))
        while i < len(modules):
            candidates: set = block_types
            j = i
            end_index = None
            while j < len(modules):
                module = modules[j][1]
                new_candidates = set()
                for candidate in candidates:
                    if isinstance(module, candidate[0]):
                        if len(candidate) == 1:
                            end_index = j
                        else:
                            new_candidates.add(candidate[1:])
                if not new_candidates:
                    break
                candidates = new_candidates
                j += 1
            if end_index is not None:
                module_list = []
                while i <= end_index:
                    module_list.append(modules[i])
                    i += 1
                blocks.append(module_list)
            else:
                i += 1
        if only_block_name:
            for idx, block in enumerate(blocks):
                blocks[idx] = [m[0] for m in block]
        return blocks
