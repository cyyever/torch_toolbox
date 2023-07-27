import os
from typing import Any

import torch
from cyy_naive_lib.log import get_logger

from ..dataset_collection import DatasetCollection
from ..dependency import has_dali, has_torchvision
from ..ml_type import DatasetType, MachineLearningPhase, ModelType

if has_dali and has_torchvision:
    from .dali_dataloader import get_dali_dataloader


def get_dataloader(
    dc: DatasetCollection,
    phase: MachineLearningPhase,
    batch_size: int,
    device: torch.device,
    cache_transforms: str | None = None,
    model_type: ModelType | None = None,
) -> Any:
    dataset = dc.get_dataset(phase=phase)
    transforms = dc.get_transforms(phase=phase)
    data_in_cpu: bool = True
    match cache_transforms:
        case "cpu" | True:
            dataset, transforms = transforms.cache_transforms(dataset=dataset)
        case "gpu" | "cuda":
            data_in_cpu = False
            dataset, transforms = transforms.cache_transforms(
                dataset=dataset, device=device
            )
    if has_dali and has_torchvision and dc.dataset_type == DatasetType.Vision:
        dataloader = get_dali_dataloader(
            dataset=dataset,
            dc=dc,
            phase=phase,
            batch_size=batch_size,
            device=device,
            model_type=model_type,
        )
        if dataloader is not None:
            return dataloader

    collate_fn = transforms.collate_batch
    kwargs: dict = {}
    use_process: bool = "USE_THREAD_DATALOADER" not in os.environ
    if use_process:
        kwargs["prefetch_factor"] = 2
        kwargs["num_workers"] = 1
        if not data_in_cpu:
            kwargs["multiprocessing_context"] = torch.multiprocessing.get_context(
                "spawn"
            )
        kwargs["persistent_workers"] = True
    else:
        get_logger().debug("use threads")
        kwargs["num_workers"] = 0
        kwargs["prefetch_factor"] = None
        kwargs["persistent_workers"] = False
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(phase == MachineLearningPhase.Training),
        pin_memory=False,
        collate_fn=collate_fn,
        generator=torch.Generator(
            device=None if "cuda" in device.type.lower() else device
        ),
        **kwargs,
    )
