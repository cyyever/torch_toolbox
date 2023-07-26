import os
from typing import Any

import torch
import torch.multiprocessing
from cyy_naive_lib.data_structure.task_queue import BatchPolicy, TaskQueue
from cyy_torch_toolbox.device import get_device_memory_info, get_devices


class CUDABatchPolicy(BatchPolicy):
    def adjust_batch_size(self, batch_size: int, **kwargs: Any) -> int:
        device = kwargs["device"]
        if (
            batch_size + 1 not in self._processing_times
            or self._processing_times[batch_size + 1]
            < self._processing_times[batch_size]
        ):
            memory_info = get_device_memory_info(device=device, consider_cache=True)
            if memory_info[device].free / memory_info[device].total > 0.2:
                return batch_size + 1
        return batch_size


class TorchTaskQueue(TaskQueue):
    def __init__(self, worker_num: int | None = None, **kwargs: Any) -> None:
        self._devices: list = get_devices()
        if worker_num is None:
            worker_num = len(self._devices)
            if "cpu" in self._devices[0].type.lower():
                worker_num = os.cpu_count()
        super().__init__(worker_num=worker_num, **kwargs)

    def _get_task_kwargs(self, worker_id: int) -> dict:
        kwargs = super()._get_task_kwargs(worker_id) | {
            "device": self._devices[worker_id % len(self._devices)]
        }
        if self._batch_process and torch.cuda.is_available():
            kwargs["batch_policy"] = CUDABatchPolicy()
        return kwargs
