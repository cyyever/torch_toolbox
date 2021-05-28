#!/usr/bin/env python3
import os
from typing import Callable

import torch.multiprocessing
from cyy_naive_lib.data_structure.task_queue import RepeatedResult, TaskQueue
from device import get_cpu_device, get_devices
from tensor import to_device


class TorchProcessTaskQueue(TaskQueue):
    def __init__(self, worker_fun: Callable = None, worker_num=None, use_manager=False):
        self.__devices = get_devices()
        if worker_num is None:
            if torch.cuda.is_available():
                worker_num = len(self.__devices)
            else:
                worker_num = os.cpu_count()
        ctx = torch.multiprocessing.get_context("spawn")
        self.__manager = None
        if use_manager:
            self.__manager = ctx.Manager()
        super().__init__(
            worker_fun=worker_fun,
            ctx=ctx,
            worker_num=worker_num,
            manager=self.__manager,
        )

    @property
    def manager(self):
        return self.__manager

    def add_task(self, task):
        if self.manager is not None:
            task = to_device(task, get_cpu_device())
        super().add_task(task)

    def put_result(self, result):
        if self.manager is not None:
            if isinstance(result, RepeatedResult):
                result.set_data(to_device(result.get_data(), get_cpu_device()))
            else:
                result = to_device(result, get_cpu_device())
        super().put_result(result)

    def __getstate__(self):
        state = super().__getstate__()
        for k in state:
            if "devices" in k:
                state[k] = None
        return state

    def _get_extra_task_arguments(self, worker_id):
        return [self.__devices[worker_id % len(self.__devices)]]
