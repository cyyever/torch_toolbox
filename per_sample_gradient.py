#!/usr/bin/env python3

import copy
import atexit
import torch

from cyy_naive_lib.list_op import split_list_to_chunks

from device import get_cuda_devices
from model_util import ModelUtil
from process_task_queue import CUDAProcessTaskQueue


def __thread_func(task, args):
    (index, input_chunk, target_chunk, loss_fun, model,) = task

    loss = None
    new_model = copy.deepcopy(model)
    device = args[0]
    new_model.to(device)
    gradient_lists = []
    for (sample_input, sample_target) in zip(input_chunk, target_chunk):
        new_model.zero_grad()
        sample_input = torch.stack([sample_input]).to(device)
        sample_target = torch.stack([sample_target]).to(device)
        loss = loss_fun(new_model(sample_input), sample_target)
        loss.backward()
        gradient_lists.append(ModelUtil(new_model).get_gradient_list())
    assert len(gradient_lists) == len(input_chunk)
    return (index, gradient_lists)


device_task_queue = None


def __exit_handler():
    global device_task_queue
    if device_task_queue is not None:
        device_task_queue.force_stop()


atexit.register(__exit_handler)


def get_per_sample_gradient(model, loss_fun, inputs, targets):
    global device_task_queue
    assert loss_fun.reduction == "mean" or loss_fun.reduction == "elementwise_mean"
    assert len(inputs) == len(targets)

    model = ModelUtil(model).deepcopy()
    if ModelUtil(model).is_pruned:
        ModelUtil(model).merge_and_remove_masks()
    model.zero_grad()
    model.share_memory()

    devices = get_cuda_devices()

    input_chunks = list(
        split_list_to_chunks(
            inputs,
            (len(inputs) +
             len(devices) -
             1) //
            len(devices)))

    target_chunks = list(
        split_list_to_chunks(
            targets,
            (len(targets) +
             len(devices) -
             1) //
            len(devices)))
    if device_task_queue is None:
        device_task_queue = CUDAProcessTaskQueue(__thread_func)
    device_task_queue.start()
    for idx, (input_chunk, target_chunk) in enumerate(
            zip(input_chunks, target_chunks)):
        device_task_queue.add_task(
            (idx, input_chunk, target_chunk, loss_fun, model,))

    gradient_dict = dict()
    for _ in range(len(input_chunks)):
        idx, gradient_list = device_task_queue.get_result()
        gradient_dict[idx] = gradient_list

    gradient_lists = []
    for idx in sorted(gradient_dict.keys()):
        gradient_lists += gradient_dict[idx]
    assert len(gradient_lists) == len(inputs)
    return gradient_lists
