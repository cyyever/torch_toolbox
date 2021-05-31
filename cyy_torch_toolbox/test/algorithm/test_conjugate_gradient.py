import torch
from device import get_device

from algorithm.conjugate_gradient import conjugate_gradient


def test_conjugate_gradient():
    device = get_device()
    a = torch.tensor([[1.0, 2.0], [2.0, 1.0]]).to(device)
    b = torch.tensor([3.0, 4.0]).to(device)
    p = conjugate_gradient(a, b)
    assert torch.all((a @ p).eq(b))
