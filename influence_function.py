#!/usr/bin/env python3

import torch

from .hessian_vector_product import hessian_vector_product
from .conjugate_gradient import conjugate_gradient_general
from .device import get_device
from .gradient import get_gradient


def get_s_test(model, loss_fun, test_instance, training_data_loader):
    device = get_device()
    test_input, test_output = test_instance
    test_input = test_input.to(device)
    test_output = test_output.to(device)
    loss = loss_fun(model(test_input), test_output)
    test_gradient = get_gradient(model, loss)
    s_test = torch.zeros(test_gradient.shape)
    s_test = s_test.to(device)
    damping = 0
    for batch in training_data_loader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        loss = loss_fun(model(inputs), targets)
        s_test += conjugate_gradient_general(
            lambda v: hessian_vector_product(model, loss, v, damping),
            test_gradient,
            epsilon=0.1,
        )
    s_test /= len(training_data_loader)
    return s_test


def get_influence_function(
    model, loss_fun, test_instance, training_data_loader, training_instance
):
    device = get_device()
    s_test = get_s_test(model, loss_fun, test_instance, training_data_loader)
    training_input, training_output = training_instance
    training_input = training_input.to(device)
    training_output = training_output.to(device)
    loss = loss_fun(model(training_input), training_output)
    training_gradient = get_gradient(model, loss)
    return -(s_test @ training_gradient)
