#!/usr/bin/env python3

import torch
from .device import get_device


def conjugate_gradient(A, b, max_iteration=None, epsilon=0.0001):
    return conjugate_gradient_general(
        lambda v: A @ v, b, max_iteration, epsilon)


def conjugate_gradient_general(
    A_product_func,
    b,
    max_iteration=None,
        epsilon=0.0001):
    x = torch.ones(b.shape)
    x = x.to(get_device())
    if max_iteration is None:
        max_iteration = b.shape[0] * 2
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


if __name__ == "__main__":
    test_A = torch.Tensor([[1, 2], [2, 1]])
    test_b = torch.Tensor([3, 4])
    test_A = test_A.to(get_device())
    test_b = test_b.to(get_device())
    p = conjugate_gradient(test_A, test_b)
    print(test_A @ p)
    print(test_b)
