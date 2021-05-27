#!/usr/bin/env python3
import os
from typing import Callable

import torch.multiprocessing
from cyy_naive_lib.data_structure.task_queue import RepeatedResult, TaskQueue
from device import get_cpu_device, get_devices
from tensor import to_device


class TorchProcessTaskQueue(TaskQueue):
    def __init__(
        self, worker_fun: Callable = None, worker_num=None, use_cpu_tensor=False
    ):
        self.devices = get_devices()
        if worker_num is None:
            if torch.cuda.is_available():
                worker_num = len(self.devices)
            else:
                worker_num = os.cpu_count()
        self.__use_cpu_tensor = use_cpu_tensor
        super().__init__(
            worker_fun=worker_fun,
            ctx=torch.multiprocessing.get_context("spawn"),
            worker_num=worker_num,
        )

    def put_result(self, result):
        if self.__use_cpu_tensor:
            if isinstance(result, RepeatedResult):
                result.set_data(to_device(result.get_data(), get_cpu_device()))
            else:
                result = to_device(result, get_cpu_device())
        super().put_result(result)

    def __getstate__(self):
        state = super().__getstate__()
        state["devices"] = None
        return state

    def _get_extra_task_arguments(self, worker_id):
        return [self.devices[worker_id % len(self.devices)]]
