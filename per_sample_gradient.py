#!/usr/bin/env python3

import copy
import threading
import torch

from cyy_naive_lib.list_op import split_list_to_chunks

from device import get_cuda_devices
from model_util import ModelUtil
from cuda_task_queue import CUDATaskQueue


def __thread_func(task, device):
    (
        idx,
        input_chunk,
        target_chunk,
        gradient_dict,
        gradient_lock,
        loss_fun,
        model,
    ) = task

    loss = None
    new_model = copy.deepcopy(model)
    new_model.to(device)
    gradient_lists = []
    for index, (sample_input, sample_target) in enumerate(
        zip(input_chunk, target_chunk)
    ):
        new_model.zero_grad()
        sample_input = torch.stack([sample_input]).to(device)
        sample_target = torch.stack([sample_target]).to(device)
        loss = loss_fun(new_model(sample_input), sample_target)
        loss.backward()
        gradient_lists.append(ModelUtil(new_model).get_gradient_list())
    assert len(gradient_lists) == len(input_chunk)
    gradient_dict[idx] = gradient_lists


def get_per_sample_gradient(model, loss_fun, inputs, targets):
    assert loss_fun.reduction == "mean" or loss_fun.reduction == "elementwise_mean"
    assert len(inputs) == len(targets)

    if ModelUtil(model).is_pruned:
        model = copy.deepcopy(model)
        ModelUtil(model).merge_and_remove_masks()
    model.zero_grad()

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

    device_task_queue = CUDATaskQueue(processor=__thread_func)
    device_task_queue.start()
    gradient_dict = dict()
    gradient_lock = threading.Lock()
    for idx, (input_chunk, target_chunk) in enumerate(
            zip(input_chunks, target_chunks)):
        device_task_queue.add_task(
            (
                idx,
                input_chunk,
                target_chunk,
                gradient_dict,
                gradient_lock,
                loss_fun,
                model,
            )
        )
    device_task_queue.stop()
    gradient_lists = []
    for idx in sorted(gradient_dict.keys()):
        gradient_lists += gradient_dict[idx]
    assert len(gradient_lists) == len(inputs)
    return gradient_lists
