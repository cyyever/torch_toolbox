#!/usr/bin/env python
from cyy_naive_lib.task_queue import TaskQueue

from device import get_cuda_devices


class CUDATaskQueue(TaskQueue):
    def __init__(self, processor):
        cuda_devices = get_cuda_devices()
        super().__init__(self, processor, len(cuda_devices))
        self.cuda_devices = cuda_devices

    def _get_extra_task_arguments(self, worker_id):
        return [self.cuda_devices[worker_id]]
