from cyy_naive_lib.data_structure.thread_context import ThreadContext
from cyy_torch_toolbox.data_structure.torch_task_queue import TorchTaskQueue


class TorchThreadTaskQueue(TorchTaskQueue):
    def __init__(self, **kwargs):
        super().__init__(mp_ctx=ThreadContext(), **kwargs)
