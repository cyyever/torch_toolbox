import os

import torch
import torch.utils.data
from cyy_naive_lib.log import get_logger

from ..data_structure.torch_process_context import TorchProcessContext
from ..dataset.collection import DatasetCollection
from ..factory import Factory
from ..hyper_parameter import HyperParameter
from ..ml_type import DatasetType, MachineLearningPhase
from ..model import ModelEvaluator

global_dataloader_factory = Factory()


def __prepare_dataloader_kwargs(
    dc: DatasetCollection,
    phase: MachineLearningPhase,
    hyper_parameter: HyperParameter,
    cache_transforms: str | None,
    device: torch.device,
    **kwargs,
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
        case "device":
            data_in_cpu = False
            assert device is not None
            dataset, transforms = transforms.cache_transforms(
                dataset=dataset, device=device
            )
        case None:
            pass
        case _:
            raise RuntimeError(cache_transforms)
    use_process: bool = "USE_THREAD_DATALOADER" not in os.environ
    if dc.dataset_type == DatasetType.Graph:
        # don't pass large graphs around processes
        use_process = False
    if cache_transforms == "device":
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
    model_evaluator: ModelEvaluator,
    **kwargs,
) -> torch.utils.data.DataLoader:
    dataloader_kwargs = __prepare_dataloader_kwargs(
        dc=dc,
        phase=phase,
        **kwargs,
    )
    constructor = global_dataloader_factory.get(dc.dataset_type)
    if constructor is not None:
        return constructor(
            dc=dc, model_evaluator=model_evaluator, phase=phase, **dataloader_kwargs
        )
    return torch.utils.data.DataLoader(**dataloader_kwargs)
