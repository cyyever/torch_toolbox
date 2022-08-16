#!/usr/bin/env python3


import torch
from cyy_naive_lib.data_structure.process_pool import ProcessPool


class TorchProcessPool(ProcessPool):
    def __init__(self, **kwargs):
        super().__init__(
            mp_context=torch.multiprocessing.get_context("spawn"), **kwargs
        )
