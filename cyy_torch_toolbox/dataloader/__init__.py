import math
import os
from typing import Any

import torch
from cyy_naive_lib.log import get_logger

from ..data_structure.torch_process_context import TorchProcessContext
from ..dataset.util import GraphDatasetUtil
from ..dataset_collection import DatasetCollection
from ..dependency import has_dali, has_torch_geometric, has_torchvision
from ..hyper_parameter import HyperParameter
from ..ml_type import DatasetType, MachineLearningPhase
from ..model_evaluator import GraphModelEvaluator, ModelEvaluator

if has_torch_geometric:
    import torch_geometric
    from torch_geometric.loader import NeighborLoader

    from .pyg_dataloader import RandomNodeLoader


if has_dali and has_torchvision:
    pass


def __prepare_dataloader_kwargs(
    dc: DatasetCollection,
    phase: MachineLearningPhase,
    hyper_parameter: HyperParameter,
    cache_transforms: str | None = None,
    device: torch.device | None = None,
) -> dict:
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
            assert device is not None
            dataset, transforms = transforms.cache_transforms(
                dataset=dataset, device=device
            )
        case _:
            if cache_transforms is not None:
                raise RuntimeError(cache_transforms)
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
    kwargs["batch_size"] = hyper_parameter.batch_size
    kwargs["shuffle"] = phase == MachineLearningPhase.Training
    kwargs["pin_memory"] = False
    kwargs["collate_fn"] = transforms.collate_batch
    kwargs["dataset"] = dataset
    return kwargs


def get_dataloader(
    dc: DatasetCollection,
    phase: MachineLearningPhase,
    hyper_parameter: HyperParameter,
    cache_transforms: str | None = None,
    device: torch.device | None = None,
    model_evaluator: ModelEvaluator | None = None,
) -> Any:
    kwargs = __prepare_dataloader_kwargs(
        dc=dc,
        phase=phase,
        hyper_parameter=hyper_parameter,
        cache_transforms=cache_transforms,
        device=device,
    )
    # if has_dali and has_torchvision and dc.dataset_type == DatasetType.Vision:
    #     dataloader = get_dali_dataloader(
    #         dataset=dataset,
    #         dc=dc,
    #         phase=phase,
    #         batch_size=hyper_parameter.batch_size,
    #         device=device,
    #         model_type=model_evaluator.model_type,
    #     )
    #     if dataloader is not None:
    #         return dataloader

    if dc.dataset_type != DatasetType.Graph:
        return torch.utils.data.DataLoader(**kwargs)
    assert has_torch_geometric
    assert isinstance(model_evaluator, GraphModelEvaluator)
    util = dc.get_dataset_util(phase=phase)
    assert isinstance(util, GraphDatasetUtil)
    assert len(util) == 1
    graph = util.get_graph(0)
    graph_dict = graph.to_dict()
    input_nodes = util.get_mask()[0]
    if hyper_parameter.extra_parameters.get("sample_neighbor", True):
        if (
            "pyg_input_nodes" in hyper_parameter.extra_parameters
            and phase in hyper_parameter.extra_parameters["pyg_input_nodes"]
        ):
            input_nodes = hyper_parameter.extra_parameters["pyg_input_nodes"][phase]
        if "batch_number" in hyper_parameter.extra_parameters:
            batch_number = hyper_parameter.extra_parameters["batch_number"]
            kwargs["batch_size"] = math.ceil(input_nodes.numel() / batch_number)
        sub_graph = type(graph)(**graph_dict)
        return NeighborLoader(
            data=sub_graph,
            num_neighbors=[hyper_parameter.extra_parameters.get("num_neighbor", 10)]
            * model_evaluator.neighbour_hop,
            input_nodes=input_nodes,
            transform=lambda data: data.to_dict(),
            **kwargs,
        )
    node_indices = torch_geometric.utils.mask_to_index(input_nodes).tolist()
    return RandomNodeLoader(node_indices, **kwargs)
