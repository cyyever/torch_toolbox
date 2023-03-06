#!/usr/bin/env python3


import torch
from cyy_naive_lib.data_structure.process_pool import ProcessPool
from cyy_naive_lib.system_info import get_operating_system


class TorchProcessPool(ProcessPool):
    def __init__(self, method: str | None = None, **kwargs) -> None:
        if method is None:
            method = "spawn" if get_operating_system() != "freebsd" else "fork"
        super().__init__(mp_context=torch.multiprocessing.get_context(method), **kwargs)
