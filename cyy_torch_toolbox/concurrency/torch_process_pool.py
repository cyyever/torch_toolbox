from typing import Any

from cyy_naive_lib.concurrency import ProcessPool

from .torch_process_context import TorchProcessContext


class TorchProcessPool(ProcessPool):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(mp_context=TorchProcessContext().get_ctx(), **kwargs)
