from typing import Callable

import torch

from ..dataset_collection import DatasetCollection
from ..factory import Factory
from ..ml_type import DatasetType
from .base import ModelEvaluator, VisionModelEvaluator

global_model_evaluator_factory = Factory()


def get_model_evaluator(
    model: torch.nn.Module, dataset_collection: DatasetCollection, **model_kwargs
) -> ModelEvaluator:
    model_evaluator_fun: Callable = ModelEvaluator
    if dataset_collection.dataset_type == DatasetType.Vision:
        model_evaluator_fun = VisionModelEvaluator
    else:
        model_evaluator_fun = global_model_evaluator_factory.get(
            dataset_collection.dataset_type
        )
    model_evaluator = model_evaluator_fun(
        model=model,
        loss_fun=model_kwargs.pop("loss_fun_name", None),
        dataset_collection=dataset_collection,
        **model_kwargs,
    )
    return model_evaluator
