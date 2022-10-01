import threading

from cyy_torch_toolbox.data_structure.torch_task_queue import TorchTaskQueue


class TorchThreadTaskQueue(TorchTaskQueue):
    def get_ctx(self):
        return threading
