#!/usr/bin/env python3
import os
from typing import Callable

import torch.multiprocessing
from cyy_naive_lib.data_structure.task_queue import TaskQueue
from cyy_torch_toolbox.device import (CudaDeviceGreedyAllocator,
                                      get_cpu_device, get_devices)
from cyy_torch_toolbox.tensor import tensor_to


class TorchProcessTaskQueue(TaskQueue):
    ctx = None
    manager = None

    def __init__(
        self,
        worker_fun: Callable = None,
        worker_num: int | None = None,
        move_data_in_cpu: bool = True,
        max_needed_cuda_bytes=None,
        use_manager: bool = True,
    ):
        self.__devices = get_devices()
        self.use_manager = use_manager
        if max_needed_cuda_bytes is not None:
            self.__devices = CudaDeviceGreedyAllocator().get_devices(
                max_needed_cuda_bytes
            )
        if worker_num is None:
            if torch.cuda.is_available():
                worker_num = len(self.__devices)
            else:
                worker_num = os.cpu_count()
        self.__move_data_in_cpu = move_data_in_cpu
        super().__init__(worker_fun=worker_fun, worker_num=worker_num)

    def get_ctx(self):
        if TorchProcessTaskQueue.ctx is None:
            TorchProcessTaskQueue.ctx = torch.multiprocessing.get_context("spawn")
        return TorchProcessTaskQueue.ctx

    def get_manager(self):
        if not self.use_manager:
            return None
        if TorchProcessTaskQueue.manager is None:
            TorchProcessTaskQueue.manager = self.get_ctx().Manager()
        return TorchProcessTaskQueue.manager

    def __getstate__(self):
        state = super().__getstate__()
        state["_TorchProcessTaskQueue__manager"] = None
        return state

    def add_task(self, task, **kwargs):
        super().add_task(self.__process_tensor(task), **kwargs)

    def put_result(self, result, **kwargs):
        super().put_result(self.__process_tensor(result), **kwargs)

    def __process_tensor(self, data):
        if self.__move_data_in_cpu:
            data = tensor_to(data, device=get_cpu_device())
        return data

    def put_data(self, data, **kwargs):
        super().put_data(self.__process_tensor(data), **kwargs)

    def _get_extra_task_arguments(self, worker_id):
        return super()._get_extra_task_arguments(worker_id) | {
            "device": self.__devices[worker_id % len(self.__devices)]
        }
