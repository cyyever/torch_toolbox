import torch.nn as nn


# def parameters_to_vector(parameters):
#     return nn.utils.parameters_to_vector(
#         [parameter.reshape(-1) for parameter in parameters]
#     )


def model_parameters_to_vector(model):
    parameters = model.parameters()
    return nn.utils.parameters_to_vector(
        [parameter.reshape(-1) for parameter in parameters]
    )


def model_gradients_to_vector(model):
    parameters = model.parameters()
    return nn.utils.parameters_to_vector(
        [parameter.grad.reshape(-1) for parameter in parameters]
    )
