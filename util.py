import torch.nn as nn


def split_list_to_chunks(my_list, chunk_size):
    return [my_list[offs: offs + chunk_size]
            for offs in range(0, len(my_list), chunk_size)]


def parameters_to_vector(parameters):
    return nn.utils.parameters_to_vector(
        [parameter.reshape(-1) for parameter in parameters]
    )


def model_parameters_to_vector(model):
    # parameters = model.parameters()
    return nn.utils.parameters_to_vector(
        [parameter.reshape(-1) for parameter in model.parameters()]
    )


def model_gradients_to_vector(model):
    return nn.utils.parameters_to_vector(
        [parameter.grad.reshape(-1) for parameter in model.parameters()]
    )
