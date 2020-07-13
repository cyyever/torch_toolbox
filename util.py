import torch.nn as nn


def parameters_to_vector(parameters):
    return nn.utils.parameters_to_vector(
        [parameter.reshape(-1) for parameter in parameters]
    )
