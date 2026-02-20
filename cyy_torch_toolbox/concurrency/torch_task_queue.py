import os
from typing import Any

import torch
from cyy_naive_lib.concurrency import BatchPolicy, TaskQueue

from ..device import DeviceGreedyAllocator, get_device_memory_info


class CUDABatchPolicy(BatchPolicy):
    def adjust_batch_size(self, batch_size: int, **kwargs: Any) -> int:
        device = kwargs["device"]
        if (
            batch_size + 1 not in self._mean_processing_times
            or self._mean_processing_times[batch_size + 1]
            < self._mean_processing_times[batch_size]
        ):
            memory_info = get_device_memory_info(device=device)
            if memory_info[device].free / memory_info[device].total > 0.2:
                return batch_size + 1
        return batch_size


class TorchTaskQueue(TaskQueue):
    def __init__(
        self,
        worker_num: int | None = None,
        batch_policy_type: type[BatchPolicy] | None = None,
        **kwargs: Any,
    ) -> None:
        self._devices: list[torch.device] = DeviceGreedyAllocator.get_devices()
        if worker_num is None:
            worker_num = len(self._devices)
            if "cpu" in self._devices[0].type.lower():
                worker_num = os.cpu_count()
        assert worker_num is not None
        if batch_policy_type is None and torch.cuda.is_available():
            batch_policy_type = CUDABatchPolicy
        super().__init__(
            worker_num=worker_num, batch_policy_type=batch_policy_type, **kwargs
        )

    def _get_task_kwargs(self, worker_id: int, use_spwan: bool) -> dict[str, Any]:
        return super()._get_task_kwargs(worker_id, use_spwan=use_spwan) | {
            "device": self._devices[worker_id % len(self._devices)]
        }
