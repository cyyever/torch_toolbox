#!/usr/bin/env python3

import atexit
import copy
import numpy as np
import torch.autograd as autograd

from cyy_naive_lib.sequence_op import split_list_to_chunks

from device import get_cuda_devices
from util import cat_tensors_to_vector
from model_util import ModelUtil
from cuda_process_task_queue import CUDAProcessTaskQueue
from model_loss import ModelWithLoss


class ModelSnapshot:
    data: dict = dict()
    prototypes: dict = dict()

    @staticmethod
    def get(model, device):
        model_class = model.__class__.__name__
        device_str = str(device)
        if model_class not in ModelSnapshot.data:
            ModelSnapshot.data[model_class] = dict()
        if device_str not in ModelSnapshot.data[model_class]:
            ModelSnapshot.data[model_class][device_str] = []
        return ModelSnapshot.data[model_class][device_str]

    @staticmethod
    def get_prototype(model):
        model_class = model.__class__.__name__
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


def __get_f(device, inputs, targets, model_with_loss, param_shape_dict):
    def f(*args):
        nonlocal inputs, targets, model_with_loss, param_shape_dict, device
        model_snapshots = ModelSnapshot.resize_and_get(
            model_with_loss.model, device, len(args)
        )
        assert len(model_snapshots) >= len(args)
        total_loss = None
        for i, arg in enumerate(args):
            cur_model_snapshot = model_snapshots[i]
            load_model_parameters(
                cur_model_snapshot, arg, param_shape_dict, device)
            cur_model_snapshot.to(device)
            loss = model_with_loss.loss_fun(cur_model_snapshot(inputs), targets)
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss
        return total_loss
    return f


def processor_fun(task, args):
    (
        idx,
        vector_chunk,
        model_with_loss,
        parameter_dict,
        input_dict,
        target_dict,
        param_shape_dict,
    ) = task
    worker_device = args[0]
    for index, vector in enumerate(vector_chunk):
        vector_chunk[index] = vector.to(worker_device)
    inputs = input_dict.get(str(worker_device))
    targets = target_dict.get(str(worker_device))
    parameter_vector = parameter_dict.get(str(worker_device))
    vector_chunk = tuple(vector_chunk)
    products = autograd.functional.vhp(
        __get_f(
            worker_device,
            inputs,
            targets,
            model_with_loss,
            param_shape_dict,
        ),
        tuple([parameter_vector] * len(vector_chunk)),
        vector_chunk,
        strict=True,
    )[1]
    return (idx, products)


task_queue = None


def __exit_handler():
    global task_queue
    if task_queue is not None:
        task_queue.force_stop()


atexit.register(__exit_handler)


def get_hessian_vector_product_func(model_with_loss: ModelWithLoss, batch):
    # get all parameters and names
    params = []
    param_shape_dict = dict()
    devices = get_cuda_devices()

    model = ModelUtil(model_with_loss.model).deepcopy()
    if ModelUtil(model).is_pruned:
        ModelUtil(model).merge_and_remove_masks()
    model.zero_grad()
    model.share_memory()

    model_snapshot = ModelSnapshot.resize_and_get(model, devices[0], 1)[0]

    for name, param in model.named_parameters():
        params.append(copy.deepcopy(param).detach())
        param_shape_dict[name] = param.shape

    parameter_snapshot = cat_tensors_to_vector(params)

    inputs_dict = dict()
    targets_dict = dict()
    parameter_dict = dict()

    for device in devices:
        inputs_dict[str(device)] = copy.deepcopy(batch[0]).to(device)
        targets_dict[str(device)] = copy.deepcopy(batch[1]).to(device)
        parameter_dict[str(device)] = copy.deepcopy(
            parameter_snapshot).to(device)

    def vhp_func(v):
        global task_queue
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

        if task_queue is None:
            task_queue = CUDAProcessTaskQueue(processor_fun)
        task_queue.start()
        for idx, vector_chunk in enumerate(vector_chunks):
            task_queue.add_task(
                (
                    idx,
                    vector_chunk,
                    ModelWithLoss(model_snapshot, model_with_loss.loss_fun),
                    parameter_dict,
                    inputs_dict,
                    targets_dict,
                    param_shape_dict,
                )
            )

        total_products = dict()
        for _ in range(len(vector_chunks)):
            idx, gradient_list = task_queue.get_result()
            total_products[idx] = gradient_list

        products = []
        for idx in sorted(total_products.keys()):
            products += total_products[idx]
        assert len(products) == len(vectors)
        if v_is_tensor:
            return products[0]
        return [p.to(devices[0]) for p in products]

    return vhp_func
