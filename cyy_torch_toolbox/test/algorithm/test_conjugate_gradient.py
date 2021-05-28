import torch
from algorithm.conjugate_gradient import conjugate_gradient
from device import get_device


def test_conjugate_gradient():
    a = torch.tensor([[1.0, 2.0], [2.0, 1.0]]).to(get_device())
    b = torch.tensor([3.0, 4.0]).to(get_device())
    p = conjugate_gradient(a, b)
    assert torch.all((a @ p).eq(b))
