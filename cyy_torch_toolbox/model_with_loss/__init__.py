import os
import sys

import torch

module_dir = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
)
if module_dir not in sys.path:
    sys.path.append(module_dir)
from cyy_torch_toolbox.dataset_collection import DatasetCollection
from cyy_torch_toolbox.dependency import has_hugging_face
from cyy_torch_toolbox.ml_type import DatasetType, ModelType

from .base import ModelEvaluator, VisionModelEvaluator
from .graph import GraphModelEvaluator
from .text import TextModelEvaluator

if has_hugging_face:
    import transformers

    from .hugging_face import HuggingFaceModelEvaluator


def get_model_with_loss(
    model: torch.nn.Module,
    dataset_collection: DatasetCollection,
    model_type: None | ModelType = None,
    model_kwargs: dict | None = None,
) -> ModelEvaluator:
    model_with_loss_fun = ModelEvaluator
    if dataset_collection.dataset_type == DatasetType.Vision:
        model_with_loss_fun = VisionModelEvaluator
    elif dataset_collection.dataset_type == DatasetType.Text:
        if has_hugging_face and isinstance(
            model, transformers.modeling_utils.PreTrainedModel
        ):
            model_with_loss_fun = HuggingFaceModelEvaluator
        else:
            model_with_loss_fun = TextModelEvaluator
    elif dataset_collection.dataset_type == DatasetType.Graph:
        model_with_loss_fun = GraphModelEvaluator
    if model_kwargs is None:
        model_kwargs = {}
    loss_fun_name = model_kwargs.get("loss_fun_name", None)
    model_type = ModelType.Classification
    model_with_loss = model_with_loss_fun(
        model=model,
        loss_fun=loss_fun_name,
        model_type=model_type,
    )
    model_path = model_kwargs.get("model_path", None)
    if model_path is not None:
        model_with_loss.model.load_state_dict(torch.load(model_path))
    word_vector_name = model_kwargs.get("word_vector_name", None)
    if word_vector_name is not None:
        from cyy_torch_toolbox.word_vector import PretrainedWordVector

        PretrainedWordVector(word_vector_name).load_to_model(
            model_with_loss=model_with_loss,
            tokenizer=dataset_collection.tokenizer,
            freeze_embedding=model_kwargs.get("freeze_word_vector", False),
        )
    return model_with_loss
