#!/usr/bin/env python3
from typing import Callable

import torch.multiprocessing
from cyy_naive_lib.data_structure.process_task_queue import ProcessTaskQueue

from device import get_cuda_devices


class CUDAProcessTaskQueue(ProcessTaskQueue):
    def __init__(self, processor_fun: Callable):
        self.cuda_devices = get_cuda_devices()
        super().__init__(
            processor_fun=processor_fun,
            ctx=torch.multiprocessing.get_context("spawn"),
            worker_num=len(self.cuda_devices),
        )

    def _get_extra_task_arguments(self, worker_id):
        return [self.cuda_devices[worker_id]]
