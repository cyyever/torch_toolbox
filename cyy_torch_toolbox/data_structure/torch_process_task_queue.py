from typing import Any

from cyy_torch_toolbox.data_structure.torch_task_queue import TorchTaskQueue

from .torch_process_context import TorchProcessContext


class TorchProcessTaskQueue(TorchTaskQueue):
    def __init__(self, use_manager: bool = True, **kwargs: Any) -> None:
        super().__init__(mp_ctx=TorchProcessContext(use_manager=use_manager), **kwargs)
