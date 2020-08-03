#!/usr/bin/env python3

import torch
from hessian_vector_product import get_hessian_vector_product_func
from conjugate_gradient import conjugate_gradient_general


def stochastic_inverse_hessian_vector_product(
    model, dataset, loss_fun, v, max_iteration=None, batch_size=1
):
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
    )
    t = 0
    cur_product = v

    if max_iteration is None:
        max_iteration = len(dataset)

    while t < max_iteration:
        for batch in data_loader:
            hvp_function = get_hessian_vector_product_func(
                model, batch, loss_fun)

            next_product = v + cur_product - hvp_function(cur_product)
            cur_product = next_product
            t += 1
            if t > max_iteration:
                return cur_product


def CG_inverse_hessian_vector_product(
        model, dataset, loss_fun, v, max_iteration=None):
    hvp_function = get_hessian_vector_product_func(model, dataset, loss_fun)
    conjugate_gradient_general(hvp_function, v, max_iteration)
