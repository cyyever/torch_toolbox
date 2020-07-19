#!/usr/bin/env python3

import threading
import copy
import numpy as np
import torch
import torch.autograd as autograd

from cyy_naive_lib.task_queue import TaskQueue
from cyy_naive_lib.list_op import split_list_to_chunks

from device import get_cuda_devices
from util import cat_tensors_to_vector
from model_util import ModelUtil


class ModelSnapshot:
    data: dict = dict()
    prototypes: dict = dict()
    lock = threading.Lock()

    @staticmethod
    def get(model, device):
        model_class = model.__class__.__name__
        device_str = str(device)
        with ModelSnapshot.lock:
            if model_class not in ModelSnapshot.data:
                ModelSnapshot.data[model_class] = dict()
            if device_str not in ModelSnapshot.data[model_class]:
                ModelSnapshot.data[model_class][device_str] = []
            return ModelSnapshot.data[model_class][device_str]

    @staticmethod
    def get_prototype(model):
        model_class = model.__class__.__name__
        with ModelSnapshot.lock:
            if model_class not in ModelSnapshot.prototypes:
                ModelSnapshot.prototypes[model_class] = copy.deepcopy(model)
            return ModelSnapshot.prototypes[model_class]

    @staticmethod
    def resize_and_get(model, device, new_number):
        snapshots = ModelSnapshot.get(model, device)
        if len(snapshots) >= new_number:
            return snapshots
        prototype = ModelSnapshot.get_prototype(model)
        while len(snapshots) < new_number:
            snapshots.append(copy.deepcopy(prototype))
        return snapshots


def load_model_parameters(model, parameters, param_shape_dict, device):
    bias = 0
    for name, shape in param_shape_dict.items():
        param_element_num = np.prod(shape)
        param = parameters.narrow(
            0, bias, param_element_num).view(
            *shape).to(device)
        ModelUtil(model).set_attr(name, param, as_parameter=False)
        bias += param_element_num
    assert bias == len(parameters)


def __get_f(
        device,
        inputs,
        targets,
        loss_fun,
        model_snapshot,
        param_shape_dict):
    def f(*args):
        nonlocal inputs, targets, loss_fun, param_shape_dict, model_snapshot, device
        model_snapshots = ModelSnapshot.resize_and_get(
            model_snapshot, device, len(args)
        )
        assert len(model_snapshots) >= len(args)
        loss = None
        for i, arg in enumerate(args):
            cur_model_snapshot = model_snapshots[i]
            load_model_parameters(
                cur_model_snapshot, arg, param_shape_dict, device)
            cur_model_snapshot.to(device)
            if loss is None:
                loss = loss_fun(cur_model_snapshot(inputs), targets)
            else:
                loss += loss_fun(cur_model_snapshot(inputs), targets)
        return loss

    return f


def __thread_func(task):
    (
        idx,
        thread_device,
        vector_chunk,
        total_products,
        product_lock,
        loss_fun,
        model_snapshot,
        parameter_vector,
        inputs,
        targets,
        param_shape_dict,
    ) = task
    for index, vector in enumerate(vector_chunk):
        vector_chunk[index] = vector.to(thread_device)
    vector_chunk = tuple(vector_chunk)
    products = autograd.functional.vhp(
        __get_f(
            thread_device,
            inputs,
            targets,
            loss_fun,
            model_snapshot,
            param_shape_dict,
        ),
        tuple(
            [parameter_vector] *
            len(vector_chunk)),
        vector_chunk,
        strict=True,
    )[1]
    with product_lock:
        total_products[idx] = products


def get_hessian_vector_product_func(model, batch, loss_fun):
    # get all parameters and names
    params = []
    param_shape_dict = dict()
    devices = get_cuda_devices()
    model_snapshot = ModelSnapshot.resize_and_get(model, devices[0], 1)[0]

    for name, param in model.named_parameters():
        params.append(copy.deepcopy(param).detach())
        param_shape_dict[name] = param.shape

    parameter_snapshot = cat_tensors_to_vector(params)

    inputs_dict = dict()
    targets_dict = dict()
    parameter_dict = dict()

    for idx, device in enumerate(devices):
        inputs_dict[idx] = copy.deepcopy(batch[0]).to(device)
        targets_dict[idx] = copy.deepcopy(batch[1]).to(device)
        parameter_dict[idx] = copy.deepcopy(parameter_snapshot).to(device)

    def vhp_func(v):
        v_is_tensor = False
        if isinstance(v, list):
            vectors = v
        elif isinstance(v, tuple):
            vectors = list(v)
        else:
            v_is_tensor = True
            vectors = [v]

        vector_chunks = list(
            split_list_to_chunks(
                vectors, (len(vectors) + len(devices) - 1) // len(devices)
            )
        )
        assert len(vector_chunks) <= len(devices)

        device_task_queue = TaskQueue(
            processor=__thread_func,
            worker_num=len(devices))
        device_task_queue.start()
        total_products = dict()
        product_lock = threading.Lock()
        for idx, vector_chunk in enumerate(vector_chunks):
            device_task_queue.add_task(
                (
                    idx,
                    devices[idx],
                    vector_chunk,
                    total_products,
                    product_lock,
                    loss_fun,
                    model_snapshot,
                    parameter_dict[idx],
                    inputs_dict[idx],
                    targets_dict[idx],
                    param_shape_dict,
                )
            )

        device_task_queue.stop()
        products = []
        for idx in sorted(total_products.keys()):
            products += total_products[idx]
        assert len(products) == len(vectors)
        if v_is_tensor:
            return products[0]
        return [p.to(devices[0]) for p in products]

    return vhp_func


if __name__ == "__main__":
    from configuration import get_task_configuration
    from cyy_naive_lib.time_counter import TimeCounter
    from cyy_naive_lib.profiling import Profile

    trainer = get_task_configuration("MNIST", True)
    training_data_loader = torch.utils.data.DataLoader(
        trainer.training_dataset, batch_size=16, shuffle=True,
    )
    parameter_vector = ModelUtil(trainer.model).get_parameter_list()
    v = torch.ones(parameter_vector.shape)
    for batch in training_data_loader:
        hvp_function = get_hessian_vector_product_func(
            trainer.model, batch, trainer.loss_fun
        )
        a = hvp_function([v, 2 * v])
        print(a)
        trainer = get_task_configuration("MNIST", True)
        hvp_function = get_hessian_vector_product_func(
            trainer.model, batch, trainer.loss_fun
        )
        a = hvp_function([v, 2 * v])
        print(a)

        with TimeCounter() as c:
            a = hvp_function(v)
            print("one use time ", c.elapsed_milliseconds())
            print(a)
            c.reset_start_time()
            a = hvp_function([v, 2 * v])
            print("two use time ", c.elapsed_milliseconds())
            print(a)
            c.reset_start_time()
            a = hvp_function([v, 2 * v, 3 * v])
            print("3 use time ", c.elapsed_milliseconds())
            print(a)
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
            # with Profile():
            #     c.reset_start_time()
            #     a = hvp_function([v] * 100)
            #     print("100 use time ", c.elapsed_milliseconds())
        break
