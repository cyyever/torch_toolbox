from shutil import which

import torch
from cyy_torch_toolbox.data_structure.torch_process_task_queue import \
    TorchProcessTaskQueue


def hello(task, args):
    assert not task
    assert args
    return {"1": torch.Tensor([1, 2, 3])}


def test_process_task_queue():
    queue = TorchProcessTaskQueue(hello, send_tensor_in_cpu=True, assemble_tensor=True)
    queue.start()
    queue.add_task(())
    res = queue.get_result()
    assert res == {"1": torch.Tensor[1, 2, 3]}
    queue.stop()
