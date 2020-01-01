#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.autograd as autograd
from device import get_device


def hessian_vector_product(model, loss, v, damping=0):
    grad = nn.utils.parameters_to_vector(
        autograd.grad(loss, model.parameters(), create_graph=True)
    )
    grad = grad.to(get_device())
    product = grad @ v
    res = nn.utils.parameters_to_vector(
        autograd.grad(product, model.parameters(), retain_graph=True)
    )
    if damping != 0:
        res += damping * v
    return res

