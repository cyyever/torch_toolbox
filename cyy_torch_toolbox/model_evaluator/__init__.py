import functools

import torch

# module_dir = os.path.realpath(
#     os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
# )
# if module_dir not in sys.path:
#     sys.path.append(module_dir)
from ..dataset_collection import DatasetCollection
from ..dependency import has_hugging_face
from ..ml_type import DatasetType, MachineLearningPhase, ModelType
from .base import ModelEvaluator, VisionModelEvaluator
from .graph import GraphModelEvaluator
from .text import TextModelEvaluator

if has_hugging_face:
    import transformers

    from .hugging_face import HuggingFaceModelEvaluator


def get_model_evaluator(
    model: torch.nn.Module,
    dataset_collection: DatasetCollection,
    model_name: str | None = None,
    model_type: None | ModelType = None,
    model_kwargs: dict | None = None,
) -> ModelEvaluator:
    model_evaluator_fun = ModelEvaluator
    if dataset_collection.dataset_type == DatasetType.Vision:
        model_evaluator_fun = VisionModelEvaluator
    elif dataset_collection.dataset_type == DatasetType.Text:
        if has_hugging_face and isinstance(model, transformers.PreTrainedModel):
            model_evaluator_fun = HuggingFaceModelEvaluator
        else:
            model_evaluator_fun = TextModelEvaluator
    elif dataset_collection.dataset_type == DatasetType.Graph:
        model_evaluator_fun = functools.partial(
            GraphModelEvaluator,
            edge_dict=dataset_collection.get_dataset_util(
                MachineLearningPhase.Training
            ).get_edge_dict(),
        )
    if model_kwargs is None:
        model_kwargs = {}
    loss_fun_name = model_kwargs.get("loss_fun_name", None)
    model_evaluator = model_evaluator_fun(
        model=model,
        model_name=model_name,
        loss_fun=loss_fun_name,
        model_type=model_type,
    )
    model_path = model_kwargs.get("model_path", None)
    if model_path is not None:
        model_evaluator.model.load_state_dict(torch.load(model_path))
    word_vector_name = model_kwargs.get("word_vector_name", None)
    if word_vector_name is not None:
        from .word_vector import PretrainedWordVector

        PretrainedWordVector(word_vector_name).load_to_model(
            model_evaluator=model_evaluator,
            tokenizer=dataset_collection.tokenizer,
            freeze_embedding=model_kwargs.get("freeze_word_vector", False),
        )
    for frozen_module_name in model_kwargs.get("frozen_module_names", []):
        model_evaluator.model_util.freeze_modules(module_name=frozen_module_name)

    return model_evaluator
