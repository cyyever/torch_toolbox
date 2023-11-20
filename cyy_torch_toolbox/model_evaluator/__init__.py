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
    model_name: str | None = None,
    model_type: None | ModelType = None,
    model_kwargs: dict | None = None,
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
    if model_kwargs is None:
        model_kwargs = {}
    model_evaluator = model_evaluator_fun(
        model=model,
        model_name=model_name,
        loss_fun=model_kwargs.get("loss_fun_name", None),
        model_type=model_type,
    )
    model_path = model_kwargs.get("model_path", None)
    if model_path is not None:
        model_evaluator.model.load_state_dict(torch.load(model_path))
    word_vector_name = model_kwargs.get("word_vector_name", None)
    if word_vector_name is not None:
        assert hasattr(dataset_collection, "tokenizer")

        PretrainedWordVector(word_vector_name).load_to_model(
            model_evaluator=model_evaluator,
            tokenizer=dataset_collection.tokenizer,
            freeze_embedding=model_kwargs.get("freeze_word_vector", False),
        )

    if "frozen_modules" in model_kwargs:
        match model_kwargs["frozen_modules"]:
            case {"types": types}:
                for t in types:
                    model_evaluator.model_util.freeze_modules(module_type=t)
            case {"names": names}:
                for name in names:
                    model_evaluator.model_util.freeze_modules(module_name=name)
            case _:
                raise NotImplementedError(model_kwargs["frozen_modules"])
    return model_evaluator
