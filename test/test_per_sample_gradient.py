#!/usr/bin/env python3

import torch
from cyy_naive_lib.time_counter import TimeCounter

from configuration import get_task_configuration
from per_sample_gradient import get_per_sample_gradient

trainer = get_task_configuration("MNIST", True)
training_data_loader = torch.utils.data.DataLoader(
    trainer.training_dataset, batch_size=64, shuffle=True,
)

cnt = 0
for batch in training_data_loader:
    with TimeCounter() as c:
        get_per_sample_gradient(
            trainer.model,
            trainer.loss_fun,
            batch[0],
            batch[1])
    cnt += 1
    if cnt > 3:
        break
