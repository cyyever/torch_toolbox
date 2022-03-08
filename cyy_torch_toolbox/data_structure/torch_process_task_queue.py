#!/usr/bin/env python3
import os
from typing import Callable

import torch.multiprocessing
from cyy_naive_lib.data_structure.task_queue import TaskQueue
from device import get_cpu_device, get_devices, put_data_to_device


class TorchProcessTaskQueue(TaskQueue):
    def __init__(
        self,
        worker_fun: Callable = None,
        worker_num: int | None = None,
        use_manager: bool = False,
        move_data_in_cpu: bool = True,
    ):
        self.__devices = get_devices()
        self.__use_manager = use_manager
        self.__manager = None
        if worker_num is None:
            if torch.cuda.is_available():
                worker_num = len(self.__devices)
            else:
                worker_num = os.cpu_count()
        self.__move_data_in_cpu = move_data_in_cpu
        super().__init__(worker_fun=worker_fun, worker_num=worker_num)

    def get_ctx(self):
        return torch.multiprocessing.get_context("spwan")

    def get_manager(self):
        if self.__use_manager:
            if self.__manager is None:
                self.__manager = self.get_ctx().Manager()
        return self.__manager

    def release(self):
        super().release()
        if self.__manager is not None:
            self.__manager.shutdown()
            self.__manager = None

    def __getstate__(self):
        state = super().__getstate__()
        state["_TorchProcessTaskQueue__manager"] = None
        return state

    def add_task(self, task, **kwargs):
        if self.__move_data_in_cpu:
            task = put_data_to_device(task, get_cpu_device())
        super().add_task(task, **kwargs)

    def put_result(self, result, **kwargs):
        if self.__move_data_in_cpu:
            result = put_data_to_device(result, get_cpu_device())
        super().put_result(result, **kwargs)

    def _get_extra_task_arguments(self, worker_id):
        return super()._get_extra_task_arguments(worker_id) | {
            "device": self.__devices[worker_id % len(self.__devices)]
        }
