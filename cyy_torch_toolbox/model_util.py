import copy
from typing import Callable, Optional, Type

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order

from tensor import cat_tensors_to_vector, load_tensor_dict


class ModelUtil:
    def __init__(self, model: torch.nn.Module):
        self.__model = model
        self.__is_pruned = None
        self.__parameter_dict: Optional[dict] = None

    @property
    def model(self):
        return self.__model

    def get_parameter_list(self, detach=True):
        return cat_tensors_to_vector(self.__get_parameter_seq(detach=detach))

    def load_parameter_list(self, parameter_list: torch.Tensor, as_parameter=True):
        parameter_dict = self.get_parameter_dict(detach=True)
        assert parameter_dict is not None
        load_tensor_dict(parameter_dict, parameter_list)
        self.load_parameter_dict(parameter_dict, as_parameter=as_parameter)

    def get_gradient_list(self):
        if self.is_pruned:
            for layer in self.model.modules():
                for name, parameter in layer.named_parameters(recurse=False):
                    if not name.endswith("_orig"):
                        assert not hasattr(layer, name + "_mask")
                        continue
                    assert parameter.grad is not None
                    real_name = name[:-5]
                    mask = getattr(layer, real_name + "_mask", None)
                    assert mask is not None
                    parameter.grad = parameter.grad * mask
        return cat_tensors_to_vector(
            (parameter.grad for parameter in self.__get_parameter_seq(detach=False))
        )

    # def deepcopy(self, keep_pruning_mask: bool = True):
    #     if self.is_pruned and not keep_pruning_mask:
    #         for layer in self.model.modules():
    #             for name, _ in layer.named_parameters(recurse=False):
    #                 if not name.endswith("_orig"):
    #                     assert not hasattr(layer, name + "_mask")
    #                     continue
    #                 real_name = name[:-5]
    #                 assert hasattr(layer, real_name + "_mask")
    #                 if hasattr(layer, real_name):
    #                     delattr(layer, real_name)
    #     return copy.deepcopy(self.model)

    def set_attr(self, name: str, value, as_parameter=True):
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
                    setattr(model, component, value)

    def get_attr(self, name: str):
        val = self.model
        components = name.split(".")
        for component in components:
            val = getattr(val, component)
        return val

    def del_attr(self, name: str):
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

    def get_original_parameters_for_pruning(self):
        res = dict()
        for layer in self.model.modules():
            for name, parameter in layer.named_parameters(recurse=False):
                real_name = name
                if name.endswith("_orig"):
                    real_name = name[:-5]
                    assert hasattr(layer, real_name + "_mask")
                else:
                    assert not hasattr(layer, real_name + "_mask")
                res[(layer, real_name)] = copy.deepcopy(parameter)
        return res

    def merge_and_remove_masks(self):
        assert self.is_pruned
        removed_items = []
        for layer in self.model.modules():
            for name, _ in layer.named_parameters(recurse=False):
                if not name.endswith("_orig"):
                    continue
                real_name = name[:-5]
                removed_items.append((layer, real_name))

        for layer, name in removed_items:
            prune.remove(layer, name)

    def change_sub_modules(self, sub_module_type: Type, f: Callable):
        changed_modules: dict = dict()
        for k, v in self.model.named_modules():
            if isinstance(v, sub_module_type):
                changed_modules[k] = f(k, v)
        for k, v in changed_modules.items():
            self.set_attr(k, v, as_parameter=False)

    def has_sub_module(self, sub_module_type: Type):
        for _, v in self.model.named_modules():
            if isinstance(v, sub_module_type):
                return True
        return False

    def get_sub_modules(self) -> list:
        modules = list(self.model.named_modules())
        result = list()
        for i, prev_module in enumerate(modules):
            is_container = False
            if isinstance(prev_module[1], torch.nn.Sequential):
                is_container = True
            if i == 0:
                is_container = True
            if not is_container:
                result.append(prev_module)
        assert result
        return result

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
            block_classes = {"Bottleneck"}
        blocks: list = []
        i = 0
        memo = set()

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

    def get_pruning_mask_list(self):
        assert self.is_pruned
        res = dict()
        for name, parameter in self.model.named_parameters():
            if name.endswith("_orig"):
                real_name = name[:-5]
                res[real_name] = self.get_attr(real_name + "_mask")
                continue
            res[name] = torch.ones_like(parameter)
        return cat_tensors_to_vector(get_mapping_values_by_key_order(res))

    def get_sparsity(self):
        parameter_list = self.get_parameter_list()
        parameter_count = len(parameter_list)
        none_zero_parameter_num = torch.sum(parameter_list != 0)
        sparsity = 100 * float(none_zero_parameter_num) / float(parameter_count)
        return (sparsity, none_zero_parameter_num, parameter_count)

    @property
    def is_pruned(self):
        if self.__is_pruned is None:
            self.__is_pruned = prune.is_pruned(self.model)
        return self.__is_pruned

    def load_parameter_dict(
        self,
        parameter_dict: dict,
        check_parameter: bool = False,
        as_parameter: bool = True,
    ):
        assert not self.is_pruned
        assert parameter_dict
        for name, parameter in parameter_dict.items():
            if check_parameter:
                assert self.has_attr(name)
            self.set_attr(name, parameter, as_parameter=as_parameter)
        if not as_parameter:
            self.__parameter_dict = parameter_dict
        else:
            self.__parameter_dict = None

    def get_parameter_dict(self, detach=True) -> dict:
        assert not self.is_pruned
        if self.__parameter_dict is not None:
            return self.__parameter_dict
        res: dict = dict()
        for name, parameter in self.model.named_parameters():
            if detach:
                parameter = parameter.detach()
            if self.is_pruned and name.endswith("_orig"):
                res[name[:-5]] = parameter
                continue
            res[name] = parameter
        self.__parameter_dict = res
        return self.__parameter_dict

    def __get_parameter_seq(self, detach=True):
        res = self.get_parameter_dict(detach=detach)
        return get_mapping_values_by_key_order(res)
