#!/usr/bin/env python3

import concurrent.futures

import torch
from cyy_naive_lib.data_structure.executor_pool import ExecutorPool


class CUDAProcessPool(ExecutorPool):
    def __init__(self):
        super().__init__(
            concurrent.futures.ProcessPoolExecutor(
                mp_context=torch.multiprocessing.get_context("spawn")
            ),
        )
