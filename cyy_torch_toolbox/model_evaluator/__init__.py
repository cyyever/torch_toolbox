import functools
from typing import Callable

import torch

from ..dataset_collection import DatasetCollection
from ..dependency import has_hugging_face, has_torch_geometric, has_torchtext
from ..ml_type import DatasetType, ModelType
from .base import ModelEvaluator, VisionModelEvaluator
from .text import TextModelEvaluator

if has_torchtext:
    from .word_vector import PretrainedWordVector

if has_hugging_face:
    import transformers

    from .hugging_face import HuggingFaceModelEvaluator

if has_torch_geometric:
    from .graph import GraphModelEvaluator


def get_model_evaluator(
    model: torch.nn.Module,
    dataset_collection: DatasetCollection,
    model_kwargs: dict,
    model_name: str | None = None,
    model_type: None | ModelType = None,
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
        model_name=model_name,
        loss_fun=model_kwargs.pop("loss_fun_name", None),
        model_type=model_type,
        **model_kwargs,
    )
    word_vector_name = model_kwargs.get("word_vector_name", None)
    if word_vector_name is not None:
        assert hasattr(dataset_collection, "tokenizer")

        PretrainedWordVector(word_vector_name).load_to_model(
            model_evaluator=model_evaluator,
            tokenizer=dataset_collection.tokenizer,
            freeze_embedding=model_kwargs.get("freeze_word_vector", False),
        )

    return model_evaluator
