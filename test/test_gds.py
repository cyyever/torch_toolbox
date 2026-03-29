import os
import sys

import pytest
import torch
from cyy_torch_toolbox.device.gds import (
    is_gds_available,
    load_state_dict,
    save_state_dict,
)

requires_gds = pytest.mark.skipif(not is_gds_available(), reason="GDS not available")
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def test_is_gds_available() -> None:
    result = is_gds_available()
    assert isinstance(result, bool)
    if sys.platform != "linux" or not torch.cuda.is_available():
        assert result is False


@requires_cuda
def test_fallback_save_load(tmp_path: object) -> None:
    path = str(tmp_path) + "/model.pt"
    model = torch.nn.Linear(8, 4).cuda()
    save_state_dict(model.state_dict(), path)
    loaded = load_state_dict(path)
    for k in model.state_dict():
        assert torch.equal(model.state_dict()[k].cpu(), loaded[k].cpu())
    os.remove(path)


@requires_gds
def test_gds_save_load() -> None:
    path = "/ssd1/gds_pytest_test.pt"
    try:
        model = torch.nn.Linear(128, 10).cuda()
        save_state_dict(model.state_dict(), path)
        loaded = load_state_dict(path, device=torch.device("cuda:0"))
        for k in model.state_dict():
            assert torch.equal(model.state_dict()[k], loaded[k])
    finally:
        if os.path.exists(path):
            os.remove(path)


@requires_gds
def test_gds_large_tensor() -> None:
    path = "/ssd1/gds_pytest_large.pt"
    try:
        sd = {"weight": torch.randn(1024, 1024, device="cuda")}
        save_state_dict(sd, path)
        loaded = load_state_dict(path, device=torch.device("cuda:0"))
        assert torch.equal(sd["weight"], loaded["weight"])
    finally:
        if os.path.exists(path):
            os.remove(path)


@requires_gds
def test_gds_multi_tensor() -> None:
    path = "/ssd1/gds_pytest_multi.pt"
    try:
        sd = {
            "a": torch.randn(256, device="cuda"),
            "b": torch.randn(64, 64, device="cuda"),
            "c": torch.randn(3, 32, 32, device="cuda"),
        }
        save_state_dict(sd, path)
        loaded = load_state_dict(path, device=torch.device("cuda:0"))
        for k in sd:
            assert torch.equal(sd[k], loaded[k])
    finally:
        if os.path.exists(path):
            os.remove(path)
