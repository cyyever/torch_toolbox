#!/usr/bin/env python3
import os
from typing import Callable

import torch.multiprocessing
from cyy_naive_lib.data_structure.task_queue import TaskQueue

from device import get_devices


class TorchProcessTaskQueue(TaskQueue):
    def __init__(self, worker_fun: Callable, worker_num=None):
        self.devices = get_devices()
        if worker_num is None:
            worker_num = len(self.devices)
            if worker_num == 1:
                worker_num = os.cpu_count()
        super().__init__(
            worker_fun=worker_fun,
            ctx=torch.multiprocessing.get_context("spawn"),
            worker_num=worker_num,
        )

    def _get_extra_task_arguments(self, worker_id):
        return [self.devices[worker_id % len(self.devices)]]
