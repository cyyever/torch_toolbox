#!/usr/bin/env python3
import os
from typing import Callable

import torch.multiprocessing
from cyy_naive_lib.data_structure.task_queue import RepeatedResult, TaskQueue
from device import get_cpu_device, get_devices, put_data_to_device


class TorchProcessTaskQueue(TaskQueue):
    def __init__(self, worker_fun: Callable = None, worker_num=None, use_manager=False):
        self.__devices = get_devices()
        self.__use_manager = use_manager
        if worker_num is None:
            if torch.cuda.is_available():
                worker_num = len(self.__devices)
            else:
                worker_num = os.cpu_count()
        ctx = torch.multiprocessing.get_context("spawn")
        manager = None
        if self.__use_manager:
            manager = ctx.Manager()
        super().__init__(
            worker_fun=worker_fun, ctx=ctx, worker_num=worker_num, manager=manager
        )

    def add_task(self, task):
        if self.__use_manager:
            task = put_data_to_device(task, get_cpu_device())
        super().add_task(task)

    def put_result(self, result, queue_name: str = "default"):
        if self.__use_manager:
            if isinstance(result, RepeatedResult):
                result.set_data(put_data_to_device(result.get_data(), get_cpu_device()))
            else:
                result = put_data_to_device(result, get_cpu_device())
        super().put_result(result, queue_name=queue_name)

    def _get_extra_task_arguments(self, worker_id):
        return [self.__devices[worker_id % len(self.__devices)]]

    def set_worker_fun(self, worker_fun, ctx=None):
        if ctx is None:
            ctx = torch.multiprocessing.get_context("spawn")
        super().set_worker_fun(worker_fun, ctx=ctx)
