#!/usr/bin/env python3


import torch
from cyy_naive_lib.data_structure.process_pool import ProcessPool


class TorchProcessPool(ProcessPool):
    def __init__(self, method="spawn", **kwargs) -> None:
        super().__init__(mp_context=torch.multiprocessing.get_context(method), **kwargs)
