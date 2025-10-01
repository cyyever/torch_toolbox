from typing import Any

import torch
from cyy_torch_toolbox import StopExecutingException
from cyy_torch_toolbox.concurrency import TorchProcessPool


def stop_training(*args, **kwargs: Any):
    raise StopExecutingException()


def train(worker_id) -> None:
    torch.ones((1, 2))


def test_process_pool() -> None:
    pool = TorchProcessPool()
    for worker_id in range(2):
        pool.submit(train, worker_id)
    pool.shutdown()
