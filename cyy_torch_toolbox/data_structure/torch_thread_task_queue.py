from cyy_naive_lib.data_structure.thread_task_queue import ThreadTaskQueue
from cyy_torch_toolbox.device import get_devices


class TorchThreadTaskQueue(ThreadTaskQueue):
    def __init__(self, *args, **kwargs):
        self.__devices = get_devices()
        super().__init__(*args, **kwargs)

    def _get_extra_task_arguments(self, worker_id):
        return super()._get_extra_task_arguments(worker_id) | {
            "device": self.__devices[worker_id % len(self.__devices)]
        }
