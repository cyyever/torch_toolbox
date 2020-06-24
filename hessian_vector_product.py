#!/usr/bin/env python3

import copy
import torch
import numpy as np
import torch.autograd as autograd

from device import get_device
from util import parameters_to_vector, model_parameters_to_vector


__cached_model_snapshots = dict()


def clear_cached_model_snapshots():
    global __cached_model_snapshots
    __cached_model_snapshots = dict()


def get_hessian_vector_product_func(model, batch, loss_fun, for_train):
    def set_attr(obj, names, val):
        if len(names) == 1:
            delattr(obj, names[0])
            setattr(obj, names[0], val)
        else:
            set_attr(getattr(obj, names[0]), names[1:], val)

    model_snapshot = copy.deepcopy(model)

    # get all parameters and names
    names = []
    params = []
    param_shapes = []
    for name, param in model_snapshot.named_parameters():
        names.append(name)
        params.append(param.detach())
        param_shapes.append(param.shape)

    device = get_device()
    inputs = batch[0].to(device)
    targets = batch[1].to(device)
    parameter_snapshot = parameters_to_vector(params).to(device)

    parameter_snapshots = list()

    def load_model_parameters(model, parameters):
        nonlocal device, names, param_shapes
        bias = 0
        for name, shape in zip(names, param_shapes):
            param_element_num = np.prod(shape)
            param = (
                parameters.narrow(
                    0, bias, param_element_num).view(
                    *shape).to(device))
            set_attr(model, name.split("."), param)
            bias += param_element_num
        assert bias == len(parameters)

    def f(*args):
        nonlocal inputs, targets, device, loss_fun, for_train, load_model_parameters
        global __cached_model_snapshots

        model_snapshots = __cached_model_snapshots[model_snapshot.__class__.__name__]
        loss = None
        for i, arg in enumerate(args):
            cur_model_snapshot = model_snapshots[i]
            load_model_parameters(cur_model_snapshot, arg)
            cur_model_snapshot.zero_grad()

            if for_train:
                cur_model_snapshot.train()
            else:
                cur_model_snapshot.eval()
            cur_model_snapshot.to(device)
            if loss is None:
                loss = loss_fun(cur_model_snapshot(inputs), targets)
            else:
                loss += loss_fun(cur_model_snapshot(inputs), targets)
        return loss

    def vhp_func(v):
        nonlocal f
        global __cached_model_snapshots

        vector_num = 1
        vectors = v
        v_is_tensor = False
        if isinstance(v, list):
            vector_num = len(v)
            vectors = tuple(v)
        elif isinstance(v, tuple):
            vector_num = len(v)
            vectors = v
        else:
            v_is_tensor = True

        while len(parameter_snapshots) < vector_num:
            parameter_snapshots.append(
                copy.deepcopy(parameter_snapshot).requires_grad_()
            )

        model_class = model_snapshot.__class__.__name__
        if model_class not in __cached_model_snapshots:
            __cached_model_snapshots[model_class] = list()
        model_snapshots = __cached_model_snapshots[model_class]

        while len(model_snapshots) < vector_num:
            model_snapshots.append(copy.deepcopy(model_snapshot))
        assert len(model_snapshots) >= vector_num
        assert len(parameter_snapshots) >= vector_num

        products = autograd.functional.vhp(
            f, tuple(parameter_snapshots[:vector_num]), vectors, strict=True
        )[1]
        if v_is_tensor:
            return products[0]
        return products

    return vhp_func


if __name__ == "__main__":
    from configuration import get_task_configuration
    from cyy_naive_lib.time_counter import TimeCounter

    trainer = get_task_configuration("MNIST", True)
    training_data_loader = torch.utils.data.DataLoader(
        trainer.training_dataset, batch_size=16, shuffle=True,
    )
    parameter_vector = model_parameters_to_vector(trainer.model)
    v = torch.ones(parameter_vector.shape)
    v = v.to(get_device())
    for batch in training_data_loader:
        hvp_function = get_hessian_vector_product_func(
            trainer.model, batch, trainer.loss_fun, True
        )
        with TimeCounter() as c:
            a = hvp_function([v, 2 * v])
            print("one use time ", c.elapsed_milliseconds())
            print(a)
            c.reset_start_time()
            a = hvp_function([v, v])
            print("one use time ", c.elapsed_milliseconds())
            c.reset_start_time()
            a = hvp_function([v, v])
            print("two use time ", c.elapsed_milliseconds())
            c.reset_start_time()
            a = hvp_function([v] * 3)
            print("3 use time ", c.elapsed_milliseconds())
            c.reset_start_time()
            a = hvp_function([v] * 4)
            print("4 use time ", c.elapsed_milliseconds())
            c.reset_start_time()
            a = hvp_function([v] * 4)
            print("4 use time ", c.elapsed_milliseconds())
            c.reset_start_time()
            a = hvp_function([v] * 10)
            print("10 use time ", c.elapsed_milliseconds())
            c.reset_start_time()
            a = hvp_function([v] * 10)
            print("10 use time ", c.elapsed_milliseconds())
            c.reset_start_time()
            a = hvp_function([v] * 100)
            print("100 use time ", c.elapsed_milliseconds())
            c.reset_start_time()
            a = hvp_function([v] * 100)
            print("100 use time ", c.elapsed_milliseconds())
        break
