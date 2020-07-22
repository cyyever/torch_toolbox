#!/usr/bin/env python3

import torch
from cyy_naive_lib.time_counter import TimeCounter
from cyy_naive_lib.profiling import Profile

from configuration import get_task_configuration
from per_sample_gradient import get_per_sample_gradient
from device import get_device
from model_util import ModelUtil


def test_get_per_sample_gradient():
    trainer = get_task_configuration("MNIST", True)
    training_data_loader = torch.utils.data.DataLoader(
        trainer.training_dataset, batch_size=64, shuffle=True,
    )

    device = get_device()
    cnt = 0
    trainer.model.to(device)
    for batch in training_data_loader:
        with TimeCounter():
            gradients = get_per_sample_gradient(
                trainer.model, trainer.loss_fun, batch[0], batch[1]
            )
            if cnt == 0:
                print("per_sample_gradient result", gradients)

        with TimeCounter():
            instance_inputs = batch[0].to(device)
            instance_targets = batch[1].to(device)
            for (instance_input, instance_target) in zip(
                instance_inputs, instance_targets
            ):
                trainer.model.zero_grad()
                output = trainer.model(torch.stack([instance_input]))
                loss = trainer.loss_fun(output, torch.stack([instance_target]))
                loss.backward()
                gradient = ModelUtil(trainer.model).get_gradient_list()
                if cnt == 0:
                    print("per_sample_gradient single gradient", gradient)

        cnt += 1
        if cnt > 3:
            break

    with Profile():
        for batch in training_data_loader:
            get_per_sample_gradient(
                trainer.model,
                trainer.loss_fun,
                batch[0],
                batch[1])
            break
    with Profile():
        for batch in training_data_loader:
            instance_inputs = batch[0].to(device)
            instance_targets = batch[1].to(device)
            for (instance_input, instance_target) in zip(
                instance_inputs, instance_targets
            ):
                trainer.model.zero_grad()
                output = trainer.model(torch.stack([instance_input]))
                loss = trainer.loss_fun(output, torch.stack([instance_target]))
                loss.backward()
                gradient = ModelUtil(trainer.model).get_gradient_list()
            break
