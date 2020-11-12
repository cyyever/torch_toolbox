#!/usr/bin/env python3

import torch
from .device import get_device


def steepest_descent(A, b, max_iteration=None, epsilon=0.0001):
    return steepest_descent_general(lambda v: A @ v, b, max_iteration, epsilon)


def steepest_descent_general(
    A_product_func,
    b,
    max_iteration=None,
        epsilon=0.0001):
    if max_iteration is None:
        max_iteration = b.shape[0]
    x = torch.ones(b.shape)
    x = x.to(get_device())
    r = b - A_product_func(x)
    delta = r @ r
    for i in range(max_iteration):
        q = A_product_func(r)
        alpha = delta / (r @ q)
        x = x + alpha * r
        if i % 50 == 0:
            r = b - A_product_func(x)
        else:
            r = r - alpha * q
        delta = r @ r
        if torch.sqrt(delta) < epsilon:
            break
    return x


if __name__ == "__main__":
    test_A = torch.Tensor([[1, 2], [2, 1]])
    test_b = torch.Tensor([3, 4])
    test_A = test_A.to(get_device())
    test_b = test_b.to(get_device())
    p = steepest_descent(test_A, test_b)
    print(test_A @ p)
    print(test_b)
