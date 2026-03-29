"""GPUDirect Storage (GDS) support for NVMe-to-GPU direct I/O."""

import os
from pathlib import Path

import torch
from cyy_naive_lib.fs.ssd import is_ssd

_gds_available: bool | None = None


def is_gds_available() -> bool:
    global _gds_available
    if _gds_available is None:
        _gds_available = False
        try:
            if torch.accelerator.is_available():
                from torch._subclasses.fake_tensor import FakeTensorMode  # noqa: F401
                from torch.cuda.gds import GdsFile  # noqa: F401

                probe = torch.empty(1, device="cuda")
                torch._C._gds_register_buffer(probe.untyped_storage())
                torch._C._gds_deregister_buffer(probe.untyped_storage())
                _gds_available = True
        except Exception:
            pass
    return _gds_available


def _use_gds(path: str | Path) -> bool:
    if not is_gds_available():
        return False
    parent = Path(path).resolve().parent
    return parent.exists() and is_ssd(parent)


def save_state_dict(state_dict: dict, path: str | Path) -> None:
    path = str(path)
    if not _use_gds(path):
        torch.save(state_dict, path)
        return

    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.cuda.gds import GdsFile
    from torch.utils.serialization import config as serialization_config

    old_alignment = serialization_config.save.storage_alignment
    try:
        serialization_config.save.storage_alignment = 4096
        with torch.serialization.skip_data():
            torch.save(state_dict, path)
    finally:
        serialization_config.save.storage_alignment = old_alignment

    with FakeTensorMode():
        offsets = {
            k: v.untyped_storage()._checkpoint_offset
            for k, v in torch.load(path, weights_only=True).items()
        }

    f = GdsFile(path, os.O_RDWR)
    try:
        for key, tensor in state_dict.items():
            f.save_storage(tensor.untyped_storage(), offsets[key])
    finally:
        del f


def load_state_dict(
    path: str | Path,
    device: torch.device | None = None,
) -> dict:
    path = str(path)
    if not _use_gds(path):
        return torch.load(path, map_location=device or "cpu", weights_only=True)

    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.cuda.gds import GdsFile

    device = device or torch.accelerator.current_accelerator()

    with FakeTensorMode():
        fake_sd = torch.load(path, weights_only=True)
    offsets = {k: v.untyped_storage()._checkpoint_offset for k, v in fake_sd.items()}

    with torch.serialization.skip_data():
        sd = torch.load(path, map_location=device, weights_only=True)

    f = GdsFile(path, os.O_RDONLY)
    try:
        for key, tensor in sd.items():
            f.load_storage(tensor.untyped_storage(), offsets[key])
    finally:
        del f
    return sd
