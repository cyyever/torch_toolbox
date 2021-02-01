#!/usr/bin/env python3

import torch
from cyy_naive_lib.time_counter import TimeCounter
from cyy_naive_lib.profiling import Profile

from configuration import get_trainer_from_configuration
from algorithm.per_sample_gradient import get_per_sample_gradient,stop_task_queue
from device import get_device


def test_get_per_sample_gradient():
    trainer = get_trainer_from_configuration("MNIST", "LeNet5")
    training_data_loader = torch.utils.data.DataLoader(
        trainer.training_dataset,
        batch_size=64,
        shuffle=True,
    )

    device = get_device()
    trainer.model.to(device)
    for cnt, batch in enumerate(training_data_loader):
        with TimeCounter():
            gradients = get_per_sample_gradient(
                trainer.model_with_loss, batch[0], batch[1]
            )
            if cnt == 0:
                print("per_sample_gradient result", gradients)
        if cnt > 3:
            break

    with Profile():
        for batch in training_data_loader:
            get_per_sample_gradient(
                trainer.model_with_loss, batch[0], batch[1])
            break
    stop_task_queue()
