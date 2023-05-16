import torch
from cyy_naive_lib.data_structure.process_pool import ProcessPool
from cyy_naive_lib.system_info import get_operating_system

from .torch_process_context import TorchProcessContext


class TorchProcessPool(ProcessPool):
    def __init__(self, **kwargs) -> None:
        super().__init__(mp_context=TorchProcessContext().get_ctx(), **kwargs)
