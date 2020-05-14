#!/usr/bin/env python3
import torch.autograd as autograd

import torch.nn as nn


def get_gradient(model, loss):
    model.zero_grad()
    return nn.utils.parameters_to_vector(
        [gradient.reshape(-1) for gradient in autograd.grad(loss, model.parameters())]
    )
