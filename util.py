import torch.nn as nn
import torch
import torch.nn.utils.prune as prune


def split_list_to_chunks(my_list, chunk_size):
    return [my_list[offs: offs + chunk_size]
            for offs in range(0, len(my_list), chunk_size)]


def parameters_to_vector(parameters):
    return nn.utils.parameters_to_vector(
        [parameter.reshape(-1) for parameter in parameters]
    )


def model_parameters_to_vector(model):
    return nn.utils.parameters_to_vector(
        [parameter.reshape(-1) for parameter in model.parameters()]
    )


def model_gradients_to_vector(model):
    return nn.utils.parameters_to_vector(
        [parameter.grad.reshape(-1) for parameter in model.parameters()]
    )


def get_pruned_parameters(model):
    if not prune.is_pruned(model):
        raise RuntimeError("not pruned model")
    parameters = dict()
    for layer in model.modules():
        for name, parameter in layer.named_parameters(recurse=False):
            if parameter is not None:
                if name.endswith("_orig"):
                    tmp_name = name[:-5]
                    if hasattr(layer, tmp_name + "_mask"):
                        name = tmp_name
                parameters[(layer, name)] = parameter
    return parameters


def get_model_sparsity(model):
    none_zero_parameter_num = 0
    parameter_count = 0
    for layer, name in get_pruned_parameters(model):
        parameter_count += len(getattr(layer, name).view(-1))
        none_zero_parameter_num += torch.sum(getattr(layer, name) != 0)
    sparsity = 100 * float(none_zero_parameter_num) / float(parameter_count)
    return (sparsity, none_zero_parameter_num, parameter_count)
