from typing import Any

import torch.multiprocessing
from cyy_naive_lib.data_structure.process_context import ProcessContext


class TorchProcessContext(ProcessContext):
    def __init__(self, **kwargs: Any) -> None:
        ctx = torch.multiprocessing
        if torch.cuda.is_available():
            ctx = torch.multiprocessing.get_context("spawn")
        super().__init__(ctx=ctx, **kwargs)
