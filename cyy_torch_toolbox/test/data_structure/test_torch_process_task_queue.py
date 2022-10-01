import torch
from cyy_torch_toolbox.data_structure.torch_process_task_queue import \
    TorchProcessTaskQueue


def hello(tasks, **kwargs):
    assert tasks == [()]
    return {"1": torch.Tensor([1, 2, 3])}


def test_process_task_queue():
    queue = TorchProcessTaskQueue(
        hello, send_tensor_in_cpu=True, assemble_tensor=True, batch_process=True
    )
    queue.start()
    queue.add_task(())
    res = queue.get_data()
    assert len(res) == 1
    assert "1" in res
    assert res["1"].tolist() == [1, 2, 3]
    queue.stop()
