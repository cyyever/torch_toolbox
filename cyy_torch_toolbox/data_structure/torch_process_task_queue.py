#!/usr/bin/env python3
import os
import time
from typing import Callable

import pynvml
import torch.multiprocessing
from cyy_naive_lib.data_structure.task_queue import TaskQueue
from cyy_naive_lib.time_counter import TimeCounter
from cyy_torch_toolbox.device import (CudaDeviceGreedyAllocator,
                                      get_cpu_device, get_device_memory_info,
                                      get_devices)
from cyy_torch_toolbox.tensor import (assemble_tensors, disassemble_tensor,
                                      tensor_to)


class CudaBatchPolicy:
    def __init__(self):
        self.__processing_times = {}
        self.__time_counter = TimeCounter()
        # pynvml.nvmlInit()
        # for device_idx in pynvml.nvmlDeviceGetCount():
        #     h = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        #     info = pynvml.nvmlDeviceGetMemoryInfo(h)
        #     self.__free_memory_dict[device] = info.free + torch.cuda.memory_reserved(
        #         device=device
        #     )
        # pynvml.nvmlShutdown()

    def start_batch(self, batch_size, **kwargs):
        self.__time_counter.reset_start_time()

    def adjust_batch_size(self, batch_size, real_batch_size, **kwargs):
        self.__processing_times[real_batch_size] = (
            self.__time_counter.elapsed_milliseconds() / real_batch_size
        )
        if real_batch_size + 1 not in self.__processing_times:
            return real_batch_size + 1
        if (
            self.__processing_times[real_batch_size + 1]
            < self.__processing_times[real_batch_size]
        ):
            return real_batch_size + 1
        return real_batch_size


class TorchProcessTaskQueue(TaskQueue):
    ctx = None
    manager = None

    def __init__(
        self,
        worker_fun: Callable = None,
        worker_num: int | None = None,
        send_tensor_in_cpu: bool = True,
        max_needed_cuda_bytes=None,
        use_manager: bool = True,
        assemble_tensor: bool = False,
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
        self.__send_tensor_in_cpu = send_tensor_in_cpu
        if assemble_tensor:
            assert send_tensor_in_cpu
        self.__assemble_tensor = assemble_tensor
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
        super().add_task(task, **kwargs)

    def put_result(self, result, **kwargs):
        super().put_result(result=self.__process_tensor(result), **kwargs)

    def __process_tensor(self, data):
        if self.__send_tensor_in_cpu:
            data = tensor_to(data, device=get_cpu_device())
            if self.__assemble_tensor:
                data = assemble_tensors(data)
        return data

    def put_data(self, data, **kwargs):
        return super().put_data(data=self.__process_tensor(data), **kwargs)

    def get_result(self, **kwargs):
        result = super().get_result(**kwargs)
        if self.__assemble_tensor:
            result = disassemble_tensor(*result)
        return result

    def get_data(self, **kwargs):
        data = super().get_data(**kwargs)
        if self.__assemble_tensor:
            data = disassemble_tensor(*data)
        return data

    def _get_task_kwargs(self, worker_id):
        return super()._get_task_kwargs(worker_id) | {
            "device": self.__devices[worker_id % len(self.__devices)]
        }
