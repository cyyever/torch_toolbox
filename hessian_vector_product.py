#!/usr/bin/env python3

import torch.autograd as autograd
from .device import get_device
from .util import parameters_to_vector


def hessian_vector_product(model, loss, v, damping=0):
    model.zero_grad()
    grad = parameters_to_vector(
        autograd.grad(loss, model.parameters(), create_graph=True)
    )
    grad = grad.to(get_device())
    product = grad @ v
    res = parameters_to_vector(
        autograd.grad(product, model.parameters(), retain_graph=True)
    )
    if damping != 0:
        res += damping * v
    return res
