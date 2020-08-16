#!/usr/bin/env python3

import torch

from cyy_naive_lib.log import get_logger

from device import get_device
from hessian_vector_product import get_hessian_vector_product_func
from conjugate_gradient import conjugate_gradient_general


def stochastic_inverse_hessian_vector_product(
    model,
    dataset,
    loss_fun,
    v,
    repeated_num=1,
    max_iteration=None,
    batch_size=1,
    dampling_term=0,
    scale=1,
    epsilon=0.0001,
):
    if max_iteration is None:
        max_iteration = len(dataset)
    get_logger().info(
        "repeated_num is %s,max_iteration is %s,batch_size is %s,dampling term is %s,scale is %s,epsilon is %s",
        repeated_num,
        max_iteration,
        batch_size,
        dampling_term,
        scale,
        epsilon,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
    )
    product_list = []
    for cur_repeated_id in range(repeated_num):
        cur_product = v
        diff = None
        iteration = 0

        epoch = 1
        looping = True
        while looping:
            for batch in data_loader:
                hvp_function = get_hessian_vector_product_func(
                    model, batch, loss_fun)

                next_product = (
                    v
                    + (1 - dampling_term) * cur_product
                    - hvp_function(cur_product).to(get_device()) / scale
                )
                diff = torch.dist(cur_product, next_product)
                get_logger().debug("diff is %s, cur repeated sequence %s, iteration is %s, max_iteration is %s", diff,cur_repeated_id,iteration,max_iteration)
                cur_product = next_product
                iteration += 1
                if (diff <= epsilon or iteration >=
                        max_iteration) and epoch > 1:
                    product_list.append(cur_product / scale)
                    looping = False
                    break
            epoch += 1
            get_logger().debug(
                "stochastic_inverse_hessian_vector_product epoch is %s", epoch
            )
    return sum(product_list) / len(product_list)


def conjugate_gradient_inverse_hessian_vector_product(
    model, dataset, loss_fun, v, max_iteration=None
):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    for batch in data_loader:
        hvp_function = get_hessian_vector_product_func(model, batch, loss_fun)
        return conjugate_gradient_general(hvp_function, v, max_iteration)
