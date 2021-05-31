#!/usr/bin/env python3

import torch


def conjugate_gradient(A, b, max_iteration=None, epsilon=0.0001):
    return conjugate_gradient_general(lambda v: A @ v, b, max_iteration, epsilon)


def conjugate_gradient_general(A_product_func, b, max_iteration=None, epsilon=0.0001):
    r"""
    Implements Conjugate Gradient illustrated by
    An Introduction to the Conjugate Gradient Method Without the Agonizing Pain
    """
    x = torch.ones(b.shape)
    x = x.to(b.device)
    if max_iteration is None:
        max_iteration = b.shape[0]
    r = b - A_product_func(x)
    d = r
    new_delta = r @ r
    for i in range(max_iteration):
        q = A_product_func(d)
        alpha = new_delta / (d @ q)
        x = x + alpha * d
        if i % 50 == 0:
            r = b - A_product_func(x)
        else:
            r = r - alpha * q
        old_delta = new_delta
        new_delta = r @ r
        d = r + (new_delta / old_delta) * d
        if torch.sqrt(new_delta) < epsilon:
            break
    return x
