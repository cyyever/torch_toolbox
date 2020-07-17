import copy
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.utils.prune as prune


from cyy_naive_lib.list_op import dict_value_by_order

import util


class ModelUtil:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.is_pruned = None

    def get_parameter_list(self):
        return util.parameters_to_vector(self.__get_parameter_seq())

    def get_gradient_list(self):
        if self.__is_pruned():
            for layer in self.model.modules():
                for name, parameter in layer.named_parameters(recurse=False):
                    if not name.endswith("_orig"):
                        assert not hasattr(layer, name + "_mask")
                        continue
                    assert parameter.grad is not None
                    real_name = name[:-5]
                    mask = getattr(layer, real_name + "_mask", None)
                    assert mask is not None
                    assert getattr(layer, real_name).grad is None
                    parameter.grad = parameter.grad * mask
        return util.parameters_to_vector(
            (parameter.grad for parameter in self.__get_parameter_seq())
        )

    def get_parameter_dict(self):
        parameter_dict = dict()
        for name, param in self.model.named_parameters():
            parameter_dict[name] = param.detach().clone()
        return parameter_dict

    def set_attr(self, name: str, value, as_parameter=True):
        model = self.model
        components = name.split(".")
        for i, component in enumerate(components):
            if i + 1 != len(components):
                model = getattr(model, component)
            else:
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

    def load_parameters(self, parameters: dict):
        for key, value in parameters.items():
            self.set_attr(key, value)

    def get_original_parameters(self):
        res = dict()
        if self.__is_pruned():
            for (layer, name, parameter, _) in self.__get_pruned_parameters():
                res[(layer, name)] = parameter
            return res
        for layer in self.model.modules():
            for name, parameter in layer.named_parameters(recurse=False):
                if parameter is None:
                    continue
                res[(layer, name)] = copy.deepcopy(parameter)
        return res

    def get_pruning_mask_list(self):
        assert self.__is_pruned()
        res = dict()
        for name, parameter in self.model.named_parameters():
            if name.endswith("_orig"):
                real_name = name[:-5]
                res[real_name] = self.get_attr(real_name + "_mask")
                continue
            res[name] = torch.ones_like(parameter)
        return util.parameters_to_vector(dict_value_by_order(res))

    def get_sparsity(self):
        none_zero_parameter_num = 0
        parameter_count = 0
        for layer in self.model.modules():
            for name, _ in layer.named_parameters(recurse=False):
                if name.endswith("_orig"):
                    name = name[:-5]
                parameter_count += len(getattr(layer, name).view(-1))
                none_zero_parameter_num += torch.sum(getattr(layer, name) != 0)
        print(none_zero_parameter_num, parameter_count)

        sparsity = 100 * float(none_zero_parameter_num) / \
            float(parameter_count)
        return (sparsity, none_zero_parameter_num, parameter_count)

    def __is_pruned(self):
        if self.is_pruned is None:
            self.is_pruned = prune.is_pruned(self.model)
        return self.is_pruned

    def __get_pruned_parameters(self):
        assert self.__is_pruned()
        res = list()
        for layer in self.model.modules():
            for name, parameter in layer.named_parameters(recurse=False):
                real_name = name
                mask = None
                if name.endswith("_orig"):
                    real_name = name[:-5]
                    mask = getattr(layer, real_name + "_mask", None)
                    assert mask is not None
                else:
                    assert not hasattr(layer, real_name + "_mask")
                    mask = torch.ones_like(parameter)
                res.append(
                    (layer,
                     real_name,
                     copy.deepcopy(parameter),
                        copy.deepcopy(mask)))
        return res

    def __get_parameter_seq(self):
        res = dict()
        for name, parameter in self.model.named_parameters():
            if self.__is_pruned() and name.endswith("_orig"):
                res[name[:-5]] = parameter
                continue
            res[name] = parameter
        return dict_value_by_order(res)
