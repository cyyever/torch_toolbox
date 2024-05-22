import torch
from cyy_torch_toolbox.concurrency.torch_process_context import \
    TorchProcessContext


def test_pipe():
    ctx = TorchProcessContext()
    p, q = ctx.create_pipe()
    p.send(torch.tensor(1))
    assert q.recv().item() == 1
