#!/usr/bin/env python3

import atexit

import torch
from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from data_structure.torch_process_task_queue import TorchProcessTaskQueue
from device import get_cuda_devices
from ml_type import MachineLearningPhase
from model_util import ModelUtil
from model_with_loss import ModelWithLoss


def __worker_fun(task, args):
    device = args[0]
    torch.cuda.set_device(device)
    (index, input_chunk, target_chunk, model_with_loss) = task

    loss = None
    gradient_lists = []
    for (sample_input, sample_target) in zip(input_chunk, target_chunk):
        model_with_loss.model.zero_grad()
        sample_input.unsqueeze_(0)
        sample_target.unsqueeze_(0)
        # we should set phase to test so that BatchNorm would use running statistics
        loss = model_with_loss(
            sample_input, sample_target, phase=MachineLearningPhase.Test, device=device
        )["loss"]
        loss.backward()
        gradient_lists.append(ModelUtil(model_with_loss.model).get_gradient_list())
    assert len(gradient_lists) == len(input_chunk)
    return (index, gradient_lists)


__task_queue = None


def stop_task_queue():
    global __task_queue
    if __task_queue is not None:
        __task_queue.force_stop()


atexit.register(stop_task_queue)


def get_sample_gradient(model_with_loss: ModelWithLoss, inputs, targets):
    global __task_queue
    assert model_with_loss.loss_fun.reduction in ("mean", "elementwise_mean")
    assert len(inputs) == len(targets)

    # model = ModelUtil(model_with_loss.model).deepcopy(keep_pruning_mask=False)
    # assert not ModelUtil(model).is_pruned
    # if ModelUtil(model).is_pruned:
    #     ModelUtil(model).merge_and_remove_masks()
    model_with_loss.model.zero_grad()
    model_with_loss.model.share_memory()

    devices = get_cuda_devices()
    master_device = devices[0]

    input_chunks = list(
        split_list_to_chunks(inputs, (len(inputs) + len(devices) - 1) // len(devices))
    )

    target_chunks = list(
        split_list_to_chunks(targets, (len(targets) + len(devices) - 1) // len(devices))
    )
    if __task_queue is None:
        __task_queue = TorchProcessTaskQueue(__worker_fun)
    __task_queue.start()
    for idx, (input_chunk, target_chunk) in enumerate(zip(input_chunks, target_chunks)):
        __task_queue.add_task(
            (
                idx,
                input_chunk,
                target_chunk,
                model_with_loss,
            )
        )

    gradient_dict = dict()
    for _ in range(len(input_chunks)):
        idx, gradient_list = __task_queue.get_result()
        gradient_dict[idx] = [p.to(master_device) for p in gradient_list]

    gradient_lists = []
    for idx in sorted(gradient_dict.keys()):
        gradient_lists += gradient_dict[idx]
    assert len(gradient_lists) == len(inputs)
    return gradient_lists
