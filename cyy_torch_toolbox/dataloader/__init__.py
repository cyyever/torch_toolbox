import os
from typing import Any

import torch
from cyy_naive_lib.log import get_logger

from ..data_structure.torch_process_context import TorchProcessContext
from ..dataset_collection import DatasetCollection
from ..dependency import has_dali, has_torch_geometric, has_torchvision
from ..ml_type import DatasetType, MachineLearningPhase, ModelType

if has_torch_geometric:
    import torch_geometric.utils

    from .pyg_dataloader import RandomNodeLoader

    # from torch_geometric.loader import RandomNodeLoader

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
    if dc.dataset_type == DatasetType.Graph:
        cache_transforms = None
    match cache_transforms:
        case "cpu":
            dataset, transforms = transforms.cache_transforms(
                dataset=dataset, device=torch.device("cpu")
            )
        case "gpu" | "cuda" | "device":
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

    kwargs: dict = {}
    use_process: bool = "USE_THREAD_DATALOADER" not in os.environ
    if dc.dataset_type == DatasetType.Graph:
        # don't pass large graphs around processes
        use_process = False
    if use_process:
        kwargs["prefetch_factor"] = 2
        kwargs["num_workers"] = 1
        if not data_in_cpu:
            kwargs["multiprocessing_context"] = TorchProcessContext().get_ctx()
        kwargs["persistent_workers"] = True
    else:
        get_logger().debug("use threads")
        kwargs["num_workers"] = 0
        kwargs["prefetch_factor"] = None
        kwargs["persistent_workers"] = False
    kwargs["batch_size"] = batch_size
    kwargs["shuffle"] = phase == MachineLearningPhase.Training
    kwargs["pin_memory"] = False

    if has_torch_geometric and dc.dataset_type == DatasetType.Graph:
        node_indices = torch_geometric.utils.mask_to_index(dataset[0]["mask"]).tolist()
        return RandomNodeLoader(node_indices, **kwargs)
    return torch.utils.data.DataLoader(
        dataset,
        collate_fn = transforms.collate_batch,
        **kwargs,
    )
