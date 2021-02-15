#!/usr/bin/env python3

import atexit

import torch.autograd as autograd
from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks

from data_structure.cuda_process_task_queue import CUDAProcessTaskQueue
from device import get_cuda_devices
from model_util import ModelUtil
from model_with_loss import ModelWithLoss


def __get_f(device, inputs, targets, model_with_loss: ModelWithLoss):
    model_util = ModelUtil(model_with_loss.model)

    def f(*args):
        nonlocal inputs, targets, model_with_loss, device, model_util
        total_loss = None
        for parameter_list in args:
            model_util.load_parameter_list(parameter_list, as_parameter=False)
            model_with_loss.model.to(device)
            loss = model_with_loss(inputs, targets)["loss"]
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss
        return total_loss

    return f


def worker_fun(task, args):
    (
        idx,
        vector_chunk,
        model_with_loss,
        parameter_dict,
        input_dict,
        target_dict,
    ) = task
    worker_device = args[0]
    for index, vector in enumerate(vector_chunk):
        vector_chunk[index] = vector.to(worker_device)
    inputs = input_dict.get(str(worker_device)).to(worker_device)
    targets = target_dict.get(str(worker_device)).to(worker_device)
    parameter_list = parameter_dict.get(str(worker_device)).to(worker_device)
    vector_chunk = tuple(vector_chunk)
    products = autograd.functional.vhp(
        __get_f(
            worker_device,
            inputs,
            targets,
            model_with_loss,
        ),
        tuple([parameter_list] * len(vector_chunk)),
        vector_chunk,
        strict=True,
    )[1]
    return (idx, products)


__task_queue = None


def stop_task_queue():
    global __task_queue
    if __task_queue is not None:
        __task_queue.force_stop()


atexit.register(stop_task_queue)


def get_hessian_vector_product_func(model_with_loss: ModelWithLoss, batch):
    devices = get_cuda_devices()

    model = ModelUtil(model_with_loss.model).deepcopy()
    model_util = ModelUtil(model)
    if model_util.is_pruned:
        model_util.merge_and_remove_masks()
    model.zero_grad()
    model.share_memory()

    parameter_list = model_util.get_parameter_list()

    inputs_dict = dict()
    targets_dict = dict()
    parameter_dict = dict()

    for device in devices:
        inputs_dict[str(device)] = batch[0]
        targets_dict[str(device)] = batch[1]
        parameter_dict[str(device)] = parameter_list

    def vhp_func(v):
        global __task_queue
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

        if __task_queue is None:
            __task_queue = CUDAProcessTaskQueue(worker_fun)
        __task_queue.start()
        for idx, vector_chunk in enumerate(vector_chunks):
            __task_queue.add_task(
                (
                    idx,
                    vector_chunk,
                    model_with_loss,
                    parameter_dict,
                    inputs_dict,
                    targets_dict,
                )
            )

        total_products = dict()
        for _ in range(len(vector_chunks)):
            idx, gradient_list = __task_queue.get_result()
            total_products[idx] = gradient_list

        products = []
        for idx in sorted(total_products.keys()):
            products += total_products[idx]
        assert len(products) == len(vectors)
        if v_is_tensor:
            return products[0]
        return [p.to(devices[0]) for p in products]

    return vhp_func
