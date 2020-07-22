#!/usr/bin/env python3

import torch
from cyy_naive_lib.time_counter import TimeCounter

from configuration import get_task_configuration
from per_sample_gradient import get_per_sample_gradient
from device import get_device
from model_util import ModelUtil


def test_get_per_sample_gradient():
    trainer = get_task_configuration("MNIST", True)
    training_data_loader = torch.utils.data.DataLoader(
        trainer.training_dataset, batch_size=64, shuffle=True,
    )

    cnt = 0
    for batch in training_data_loader:
        with TimeCounter():
            get_per_sample_gradient(
                trainer.model,
                trainer.loss_fun,
                batch[0],
                batch[1])
        cnt += 1
        if cnt > 3:
            break


def test_per_sample_gradient_simple():
    trainer = get_task_configuration("MNIST", True)
    training_data_loader = torch.utils.data.DataLoader(
        trainer.training_dataset, batch_size=64, shuffle=True,
    )

    device = get_device()
    cnt = 0
    trainer.model.to(device)
    for batch in training_data_loader:
        with TimeCounter():
            instance_inputs = batch[0].to(device)
            instance_targets = batch[1].to(device)
            for i in range(len(instance_inputs)):
                instance_input = instance_inputs[i]
                instance_target = instance_targets[i]
                output = trainer.model(torch.stack([instance_input]))
                loss = trainer.loss_fun(output, torch.stack([instance_target]))
                loss.backward()
                ModelUtil(trainer.model).get_gradient_list()
        cnt += 1
        if cnt > 3:
            break
