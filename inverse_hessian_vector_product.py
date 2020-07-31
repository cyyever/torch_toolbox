#!/usr/bin/env python3

import torch
from hessian_vector_product import get_hessian_vector_product_func


def stochastic_inverse_hessian_vector_product(
    model, dataset, loss_fun, v, loop=None, batch_size=1
):
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
    )
    t = 0
    cur_product = v

    if loop is None:
        loop = len(dataset)

    while t < loop:
        for batch in data_loader:
            hvp_function = get_hessian_vector_product_func(
                model, batch, loss_fun)

            print("hvp_function(cur_product) is ", hvp_function(cur_product))
            cur_product = v + cur_product - hvp_function(cur_product)
            print("cur_product is ", cur_product)
            t += 1
            if t > loop:
                return cur_product
