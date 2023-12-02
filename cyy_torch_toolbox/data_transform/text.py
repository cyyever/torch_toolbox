import functools

import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.dataset_collection import DatasetCollection
from cyy_torch_toolbox.dependency import has_hugging_face, has_spacy
from cyy_torch_toolbox.ml_type import (DatasetType, MachineLearningPhase,
                                       ModelType, TransformType)
from cyy_torch_toolbox.model_evaluator.text import TextModelEvaluator

from .common import backup_target, int_target_to_text, replace_str
from .template import get_text_template, interpret_template

if has_spacy:
    from cyy_torch_toolbox.tokenizer.spacy import SpacyTokenizer

if has_hugging_face:
    import transformers as hugging_face_transformers


def add_text_extraction(dc: DatasetCollection) -> None:
    assert dc.dataset_type == DatasetType.Text
    # ExtractData
    dc.append_transform(backup_target, key=TransformType.ExtractData)


def __apply_tokenizer_transforms(
    dc: DatasetCollection,
    model_evaluator: TextModelEvaluator,
    max_len: int | None,
    for_input: bool,
) -> None:
    if for_input:
        batch_key = TransformType.InputBatch
    else:
        batch_key = TransformType.TargetBatch
    match model_evaluator.tokenizer:
        case SpacyTokenizer():
            if for_input:
                key = TransformType.Input
            else:
                key = TransformType.Target
            dc.append_transform(model_evaluator.tokenizer, key=key)
            if max_len is not None:
                dc.append_transform(
                    lambda a: a[:max_len],
                    key=key,
                )
            dc.append_transform(torch.LongTensor, key=key)
            dc.append_transform(
                functools.partial(
                    torch.nn.utils.rnn.pad_sequence,
                    padding_value=model_evaluator.tokenizer.get_index("<pad>"),
                ),
                key=batch_key,
            )
        case hugging_face_transformers.PreTrainedTokenizerBase():
            dc.append_transform(
                functools.partial(
                    model_evaluator.tokenizer,
                    max_length=max_len,
                    padding="max_length",
                    return_tensors="pt",
                    truncation=True,
                ),
                key=batch_key,
            )
        case _:
            raise NotImplementedError(type(model_evaluator.tokenizer))


def get_label_to_text_mapping(dataset_name: str) -> dict | None:
    match dataset_name.lower():
        case "multi_nli":
            return {0: "entailment", 1: "neutral", 2: "contradiction"}
        case "imdb":
            return {0: "negative", 1: "positive"}
    return None


def add_text_transforms(
    dc: DatasetCollection, model_evaluator: TextModelEvaluator
) -> None:
    assert dc.dataset_type == DatasetType.Text
    dataset_name: str = dc.name.lower()
    # InputText
    if dataset_name == "imdb":
        dc.append_transform(
            functools.partial(replace_str, old="<br />", new=""),
            key=TransformType.InputText,
        )

    assert model_evaluator.model_type is not None
    text_template = get_text_template(
        dataset_name=dataset_name, model_type=model_evaluator.model_type
    )
    if text_template is not None:
        dc.append_transform(
            functools.partial(interpret_template, template=text_template),
            key=TransformType.InputText,
        )

    # Input && InputBatch
    input_max_len = dc.dataset_kwargs.get("max_len", None)
    if input_max_len is None:
        input_max_len = dc.dataset_kwargs.get("input_max_len", None)
    get_logger().info("use input text max_len %s", input_max_len)
    __apply_tokenizer_transforms(
        dc=dc, model_evaluator=model_evaluator, max_len=input_max_len, for_input=True
    )

    # Target
    if (
        model_evaluator.model_type == ModelType.TextGeneration
        or model_evaluator.get_underlying_model_type() == ModelType.TextGeneration
    ):
        mapping = get_label_to_text_mapping(dataset_name)
        if mapping is not None:
            dc.append_transform(
                functools.partial(int_target_to_text, mapping=mapping),
                key=TransformType.Target,
            )
        elif isinstance(
            dc.get_dataset_util(phase=MachineLearningPhase.Training).get_sample_label(
                0
            ),
            int,
        ):
            dc.append_transform(int_target_to_text, key=TransformType.Target)
        max_len = dc.dataset_kwargs.get("output_max_len", None)
        get_logger().info("use output text max len %s", max_len)
        __apply_tokenizer_transforms(
            dc=dc, model_evaluator=model_evaluator, max_len=max_len, for_input=False
        )
    # elif model_evaluator.model_type == ModelType.Classification:
    #     if isinstance(
    #         dc.get_dataset_util(phase=MachineLearningPhase.Training).get_sample_label(
    #             0
    #         ),
    #         str,
    #     ):
    #         label_names = dc.get_label_names()
    #         dc.append_transform(
    #             str_target_to_int(label_names), key=TransformType.Target
    #         )
