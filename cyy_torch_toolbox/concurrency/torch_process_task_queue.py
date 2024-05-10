from typing import Any

from .torch_process_context import TorchProcessContext
from .torch_task_queue import TorchTaskQueue


class TorchProcessTaskQueue(TorchTaskQueue):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(mp_ctx=TorchProcessContext(), **kwargs)
