from cyy_torch_toolbox.data_structure.torch_process_context import \
    TorchProcessContext


def test_pipe():
    ctx = TorchProcessContext()
    p, q = ctx.create_pipe()
    p.send(1)
    assert q.recv() == 1
