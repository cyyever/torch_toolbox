import torch
from cyy_torch_toolbox.concurrency import TorchProcessTaskQueue


def hello(tasks, **kwargs):
    assert tasks == [()]
    return {"1": torch.Tensor([1, 2, 3])}


def test_process_task_queue() -> None:
    queue = TorchProcessTaskQueue(
        batch_process=True,
    )
    queue.start(worker_fun=hello)
    queue.add_task(())
    res = queue.get_data()
    assert res is not None
    data = res[0]
    assert isinstance(data, dict)
    assert len(data) == 1
    assert "1" in data
    assert data["1"].tolist() == [1, 2, 3]
    queue.stop()
