#!/usr/bin/env python3
import torch.autograd as autograd

from .util import parameters_to_vector


def get_gradient(model, loss_fun):
    return parameters_to_vector(autograd.grad(loss_fun, model.parameters()))

