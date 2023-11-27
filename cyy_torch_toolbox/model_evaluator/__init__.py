import functools
from typing import Callable

import torch

from ..dataset_collection import DatasetCollection
from ..dependency import has_hugging_face, has_torch_geometric
from ..ml_type import DatasetType
from .base import ModelEvaluator, VisionModelEvaluator
from .text import TextModelEvaluator

if has_hugging_face:
    import transformers

    from .hugging_face import HuggingFaceModelEvaluator

if has_torch_geometric:
    from .graph import GraphModelEvaluator


def get_model_evaluator(
    model: torch.nn.Module, dataset_collection: DatasetCollection, **model_kwargs
) -> ModelEvaluator:
    model_evaluator_fun: Callable = ModelEvaluator
    if dataset_collection.dataset_type == DatasetType.Vision:
        model_evaluator_fun = VisionModelEvaluator
    elif dataset_collection.dataset_type == DatasetType.Text:
        if has_hugging_face and isinstance(model, transformers.PreTrainedModel):
            model_evaluator_fun = HuggingFaceModelEvaluator
        else:
            model_evaluator_fun = TextModelEvaluator
    elif dataset_collection.dataset_type == DatasetType.Graph:
        model_evaluator_fun = functools.partial(GraphModelEvaluator, dataset_collection)
    model_evaluator = model_evaluator_fun(
        model=model,
        loss_fun=model_kwargs.pop("loss_fun_name", None),
        **model_kwargs,
    )
    return model_evaluator
