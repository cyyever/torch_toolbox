import torch
from cyy_torch_toolbox.data_structure.torch_process_pool import \
    TorchProcessPool
from cyy_torch_toolbox.ml_type import StopExecutingException


def stop_training(*args, **kwargs):
    raise StopExecutingException()


def train(worker_id) -> None:
    torch.ones((1, 2))


def test_process_pool() -> None:
    pool = TorchProcessPool()
    for worker_id in range(2):
        pool.submit(train, worker_id)
    pool.shutdown()
