import os
from typing import Any

import torch
import torch.multiprocessing
from cyy_naive_lib.concurrency import ProcessContext


class TorchProcessContext(ProcessContext):
    def __init__(self, **kwargs: Any) -> None:
        ctx: Any = torch.multiprocessing
        if torch.accelerator.is_available():
            ctx = torch.multiprocessing.get_context(
                os.getenv("CYY_TORCH_MP_CTX", "spawn")
            )
        super().__init__(ctx=ctx, **kwargs)
