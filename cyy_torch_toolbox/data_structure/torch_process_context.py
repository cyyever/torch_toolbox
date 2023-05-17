from typing import Any

import torch.multiprocessing
from cyy_naive_lib.data_structure.process_context import ProcessContext
from cyy_naive_lib.system_info import get_operating_system


class TorchProcessContext(ProcessContext):
    def __init__(self, **kwargs: Any) -> None:
        method: str = "spawn" if get_operating_system() != "freebsd" else "fork"
        ctx = torch.multiprocessing.get_context(method)
        super().__init__(ctx=ctx, **kwargs)
