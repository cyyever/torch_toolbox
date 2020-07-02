#!/usr/bin/env python3

import copy
import numpy as np
import torch
import torch.autograd as autograd

from device import get_device
from util import (
    parameters_to_vector,
    model_parameters_to_vector,
    set_model_attr,
    del_model_attr,
)


__cached_model_snapshots = dict()
__cached_parameter_vectors = dict()


class ModelSnapshot:
    data = dict()

    @staticmethod
    def clear(model_class):
        ModelSnapshot.data.pop(model_class, None)

    @staticmethod
    def get(model_class, device):
        device_str = str(device)
        if model_class not in ModelSnapshot.data:
            ModelSnapshot.data[model_class] = dict()
        if device_str not in ModelSnapshot.data[model_class]:
            ModelSnapshot.data[model_class][device_str] = []
        return ModelSnapshot.data[model_class][device_str]

    @staticmethod
    def resize(model, device, new_number):
        model_class = model.__class__.__name__
        snapshots = ModelSnapshot.get(model_class, device)
        while len(snapshots) < new_number:
            snapshots.append(copy.deepcopy(model))


def load_model_parameters(model, parameters, param_shape_dict, device):
    bias = 0
    for name, shape in param_shape_dict.items():
        param_element_num = np.prod(shape)
        param = parameters.narrow(
            0, bias, param_element_num).view(
            *shape).to(device)
        set_model_attr(model, name.split("."), param, as_parameter=False)
        bias += param_element_num
    assert bias == len(parameters)


def get_hessian_vector_product_func(model, batch, loss_fun, for_train):
    # get all parameters and names
    params = []
    param_shape_dict = dict()
    model_snapshot = copy.deepcopy(model)

    for name, param in model_snapshot.named_parameters():
        params.append(param.detach())
        param_shape_dict[name] = param.shape

    for name in param_shape_dict:
        del_model_attr(model_snapshot, name.split("."))

    # device = torch.device("cuda:2")
    # device2 = torch.device("cuda:2")
    device = torch.device("cpu")
    device2 = torch.device("cpu")
    assert device == device2
    # device = get_device()
    inputs = batch[0].to(device)
    targets = batch[1].to(device)
    parameter_snapshot = parameters_to_vector(params)

    def f(*args):
        nonlocal inputs, targets, device, loss_fun, for_train, param_shape_dict, model_snapshot
        model_class = model_snapshot.__class__.__name__
        ModelSnapshot.resize(model_snapshot, device, len(args))
        model_snapshots = ModelSnapshot.get(model_class, device)
        assert len(model_snapshots) >= len(args)
        loss = None
        for i, arg in enumerate(args):
            cur_model_snapshot = model_snapshots[i]
            load_model_parameters(
                cur_model_snapshot, arg, param_shape_dict, device)
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
        nonlocal f, device

        v_is_tensor = False
        if isinstance(v, list):
            vector_num = len(v)
            vectors = v
        elif isinstance(v, tuple):
            vector_num = len(v)
            vectors = list(v)
        else:
            v_is_tensor = True
            vector_num = 1
            vectors = [v]

        for index, vector in enumerate(vectors):
            vectors[index] = vector.to(device)
        vectors = tuple(vectors)

        products = autograd.functional.vhp(
            f, tuple([parameter_snapshot] * vector_num), vectors, strict=True
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
    for batch in training_data_loader:
        hvp_function = get_hessian_vector_product_func(
            trainer.model, batch, trainer.loss_fun, True
        )
        with TimeCounter() as c:
            a = hvp_function(v)
            print("one use time ", c.elapsed_milliseconds())
            print(a)
            c.reset_start_time()
            a = hvp_function([v, 2 * v])
            print("two use time ", c.elapsed_milliseconds())
            print(a)
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
