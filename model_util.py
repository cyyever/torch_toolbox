import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import util


class ModelUtil:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def get_parameter_list(self):
        return util.parameters_to_vector(self.model.parameters())

    def get_gradient_list(self):
        return util.parameters_to_vector(
            [parameter.grad for parameter in self.model.parameters()]
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

    def get_pruned_parameters(self):
        res = list()
        for layer in self.model.modules():
            for name, parameter in layer.named_parameters(recurse=False):
                if parameter is None:
                    continue
                assert name.endswith("_orig")
                real_name = name[:-5]
                mask = getattr(layer, real_name + "_mask", None)
                assert mask
                res.append(
                    (layer,
                     real_name,
                     copy.deepcopy(parameter),
                        copy.deepcopy(mask)))
        return res

    def get_original_parameters(self):
        res = dict()
        if prune.is_pruned(self.model):
            for (layer, name, parameter, _) in self.get_pruned_parameters():
                res[(layer, name)] = parameter
            return res
        for layer in self.model.modules():
            for name, parameter in layer.named_parameters(recurse=False):
                if parameter is None:
                    continue
                res[(layer, name)] = copy.deepcopy(parameter)
        return res

    def get_pruning_mask(self):
        pruned_parameters = self.get_pruned_parameters()
        if not pruned_parameters:
            raise RuntimeError("not pruned")
        return util.parameters_to_vector((v[3] for v in pruned_parameters))

    def merge_and_remove_pruning_mask(self):
        if not prune.is_pruned(self.model):
            return
        for layer in self.model.modules():
            for name, parameter in layer.named_parameters(recurse=False):
                if parameter is None:
                    continue
                assert name.endswith("_orig")
                real_name = name[:-5]
                mask = getattr(layer, real_name + "_mask")
                orig = getattr(layer, real_name + "_orig")
                delattr(layer, real_name + "_orig")
                delattr(layer, real_name + "_mask")
                delattr(layer, real_name)
                layer.register_parameter(
                    real_name, torch.nn.Parameter(mask * orig),
                )

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
