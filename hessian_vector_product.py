#!/usr/bin/env python3

import copy
import numpy as np
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


def get_hessian_vector_product_func(model, batch, loss_fun, for_train):
    cur_model = copy.deepcopy(model)
    cur_model.zero_grad()
    if for_train:
        cur_model.train()
    else:
        cur_model.eval()

    device = get_device()
    cur_model.to(device)
    inputs = batch[0].to(device)
    targets = batch[1].to(device)

    # def del_attr(obj, names):
    #     if len(names) == 1:
    #         delattr(obj, names[0])
    #     else:
    #         del_attr(getattr(obj, names[0]), names[1:])

    # get all parameters and names
    names = []
    params = []
    param_shapes = []
    for name, param in cur_model.named_parameters():
        names.append(name)
        params.append(param.detach().requires_grad_())
        param_shapes.append(param.shape)

    parameter_vector = parameters_to_vector(params)

    # for name in names:
    #     del_attr(cur_model, name.split("."))

    inputs = batch[0].to(device)
    targets = batch[1].to(device)

    def set_attr(obj, names, val):
        if len(names) == 1:
            delattr(obj, names[0])
            setattr(obj, names[0], val)
        else:
            set_attr(getattr(obj, names[0]), names[1:], val)

    def f(x):
        nonlocal inputs
        nonlocal targets
        nonlocal loss_fun, names, params, cur_model, param_shapes

        bias = 0
        for name, shape in zip(names, param_shapes):
            param_element_num = np.prod(shape)
            param = x.narrow(0, bias, param_element_num).view(*shape)
            set_attr(cur_model, name.split("."), param)
            bias += param_element_num

        assert bias == len(parameter_vector)
        return loss_fun(cur_model(inputs), targets)

    def vhp_func(v):
        nonlocal f, parameter_vector
        return autograd.functional.vhp(f, parameter_vector, v, strict=True)[1]

    return vhp_func
