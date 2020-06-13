import os
import PIL
import torch.nn as nn
import torch
import torch.nn.utils.prune as prune
import torchvision


def split_list_to_chunks(my_list, chunk_size):
    return [my_list[offs: offs + chunk_size]
            for offs in range(0, len(my_list), chunk_size)]


def parameters_to_vector(parameters):
    return nn.utils.parameters_to_vector(
        [parameter.reshape(-1) for parameter in parameters]
    )


def model_parameters_to_vector(model):
    return parameters_to_vector(model.parameters())


def model_gradients_to_vector(model):
    return parameters_to_vector(
        [parameter.grad for parameter in model.parameters()])


def get_pruned_parameters(model):
    parameters = dict()
    for layer_index, layer in enumerate(model.modules()):
        for name, parameter in layer.named_parameters(recurse=False):
            if parameter is None:
                continue
            mask = None
            if name.endswith("_orig"):
                tmp_name = name[:-5]
                mask = getattr(layer, tmp_name + "_mask", None)
                if mask is not None:
                    name = tmp_name
            parameters[(layer, name)] = (parameter, mask, layer_index)
    return parameters


def get_pruning_mask(model):
    if not prune.is_pruned(model):
        raise RuntimeError("not pruned model")
    return parameters_to_vector(
        [v[1] for v in get_pruned_parameters(model).values()])


def get_model_sparsity(model):
    none_zero_parameter_num = 0
    parameter_count = 0
    for layer, name in get_pruned_parameters(model):
        parameter_count += len(getattr(layer, name).view(-1))
        none_zero_parameter_num += torch.sum(getattr(layer, name) != 0)
    sparsity = 100 * float(none_zero_parameter_num) / float(parameter_count)
    return (sparsity, none_zero_parameter_num, parameter_count)


def save_sample(dataset, idx, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(dataset[idx][0], PIL.Image.Image):
        dataset[idx][0].save(path)
        return
    torchvision.utils.save_image(dataset[idx][0], path)
