import torch
from algorithm.conjugate_gradient import conjugate_gradient
from device import get_device


def test_conjugate_gradient():
    a = torch.Tensor([[1, 2], [2, 1]]).to(get_device())
    b = torch.Tensor([3, 4]).to(get_device())
    p = conjugate_gradient(a, b)
    assert torch.all((a @ p).eq(b))
