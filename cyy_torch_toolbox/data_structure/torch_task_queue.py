#!/usr/bin/env python3
import os

import torch
import torch.multiprocessing
from cyy_naive_lib.data_structure.task_queue import TaskQueue
from cyy_naive_lib.time_counter import TimeCounter
from cyy_torch_toolbox.device import (CudaDeviceGreedyAllocator,
                                      get_cuda_device_memory_info, get_devices)


class CudaBatchPolicy:
    def __init__(self):
        self.__processing_times = {}
        self.__time_counter = TimeCounter()

    def start_batch(self, **kwargs):
        self.__time_counter.reset_start_time()

    def end_batch(self, batch_size, **kwargs):
        self.__processing_times[batch_size] = (
            self.__time_counter.elapsed_milliseconds() / batch_size
        )

    def adjust_batch_size(self, batch_size, **kwargs):
        if (
            batch_size + 1 not in self.__processing_times
            or self.__processing_times[batch_size + 1]
            < self.__processing_times[batch_size]
        ):
            memory_info = get_cuda_device_memory_info(consider_cache=True)
            current_device_idx = torch.cuda.current_device()

            if (
                memory_info[current_device_idx].free
                / memory_info[current_device_idx].total
                > 0.2
            ):
                return batch_size + 1
        return batch_size


class TorchTaskQueue(TaskQueue):
    def __init__(
        self, max_needed_cuda_bytes=None, worker_num: int | None = None, **kwargs
    ):
        if max_needed_cuda_bytes is not None:
            self._devices = CudaDeviceGreedyAllocator().get_devices(
                max_needed_cuda_bytes
            )
        else:
            self._devices = get_devices()
        if worker_num is None:
            if torch.cuda.is_available():
                worker_num = len(self._devices)
            else:
                worker_num = os.cpu_count()
        super().__init__(worker_num=worker_num, **kwargs)

    def _get_task_kwargs(self, worker_id) -> dict:
        kwargs = super()._get_task_kwargs(worker_id) | {
            "device": self._devices[worker_id % len(self._devices)]
        }
        if self._batch_process:
            if torch.cuda.is_available():
                kwargs["batch_policy"] = CudaBatchPolicy()
        return kwargs
