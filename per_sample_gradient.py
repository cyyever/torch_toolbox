#!/usr/bin/env python3

import copy
import atexit
import torch

from cyy_naive_lib.sequence_op import split_list_to_chunks

from device import get_cuda_devices
from model_util import ModelUtil
from model_loss import ModelWithLoss
from process_task_queue import CUDAProcessTaskQueue


def __worker_fun(task, args):
    (index, input_chunk, target_chunk, model_with_loss, master_device) = task

    loss = None
    device = args[0]
    model_with_loss.model.to(device)
    gradient_lists = []
    for (sample_input, sample_target) in zip(input_chunk, target_chunk):
        model_with_loss.model.zero_grad()
        sample_input = torch.stack([sample_input]).to(device)
        sample_target = torch.stack([sample_target]).to(device)
        loss = model_with_loss(sample_input, sample_target)
        loss.backward()
        gradient_lists.append(
            ModelUtil(
                model_with_loss.model).get_gradient_list().to(master_device))
    assert len(gradient_lists) == len(input_chunk)
    return (index, gradient_lists)


__task_queue = None


def __exit_handler():
    global __task_queue
    if __task_queue is not None:
        __task_queue.force_stop()


atexit.register(__exit_handler)


def get_per_sample_gradient(model_with_loss: ModelWithLoss, inputs, targets):
    global __task_queue
    assert model_with_loss.get_loss_fun().reduction in ("mean", "elementwise_mean")
    assert len(inputs) == len(targets)

    model = ModelUtil(model_with_loss.model).deepcopy(keep_pruning_mask=False)
    assert not ModelUtil(model).is_pruned
    # if ModelUtil(model).is_pruned:
    #     ModelUtil(model).merge_and_remove_masks()
    model.zero_grad()
    model.share_memory()

    devices = get_cuda_devices()
    master_device = devices[0]

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
    if __task_queue is None:
        __task_queue = CUDAProcessTaskQueue(__worker_fun)
    __task_queue.start()
    for idx, (input_chunk, target_chunk) in enumerate(
            zip(input_chunks, target_chunks)):
        __task_queue.add_task(
            (idx,
             input_chunk,
             target_chunk,
             ModelWithLoss(
                 copy.deepcopy(model),
                 model_with_loss.get_loss_fun()),
                master_device,
             ))

    gradient_dict = dict()
    for _ in range(len(input_chunks)):
        idx, gradient_list = __task_queue.get_result()
        gradient_dict[idx] = gradient_list

    gradient_lists = []
    for idx in sorted(gradient_dict.keys()):
        gradient_lists += gradient_dict[idx]
    assert len(gradient_lists) == len(inputs)
    return gradient_lists
