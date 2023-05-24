import functools

import torch
from cyy_naive_lib.log import get_logger

from ..dataset_collection import (ClassificationDatasetCollection,
                                  DatasetCollection)
from ..dependency import has_hugging_face, has_spacy, has_torchtext
from ..ml_type import (DatasetType, MachineLearningPhase, ModelType,
                       TransformType)
from ..model_evaluator import ModelEvaluator
from .common import (int_target_to_text, replace_str, str_target_to_int,
                     swap_input_and_target, target_offset)
from .template import get_text_template, interpret_template

if has_torchtext:
    import torchtext
if has_spacy:
    from ..tokenizer.spacy import SpacyTokenizer

if has_hugging_face:
    import transformers as hugging_face_transformers


def add_text_extraction(dc: DatasetCollection) -> None:
    assert dc.dataset_type == DatasetType.Text
    assert has_torchtext
    # ExtractData
    match dc.name.lower():
        case "imdb":
            dc.append_transform(swap_input_and_target, key=TransformType.ExtractData)


def __apply_tokenizer_transforms(
    dc: DatasetCollection, max_len: int | None, for_input: bool
) -> None:
    if for_input:
        batch_key = TransformType.InputBatch
    else:
        batch_key = TransformType.TargetBatch
    match dc.tokenizer:
        case SpacyTokenizer():
            if for_input:
                key = TransformType.Input
            else:
                key = TransformType.Target
            dc.append_transform(dc.tokenizer, key=key)
            if max_len is not None:
                dc.append_transform(
                    torchtext.transforms.Truncate(max_seq_len=max_len),
                    key=key,
                )
            dc.append_transform(torch.LongTensor, key=key)
            dc.append_transform(
                functools.partial(
                    torch.nn.utils.rnn.pad_sequence,
                    padding_value=dc.tokenizer.vocab["<pad>"],
                ),
                key=batch_key,
            )
        case hugging_face_transformers.PreTrainedTokenizerBase():
            dc.append_transform(
                functools.partial(
                    dc.tokenizer,
                    max_length=max_len,
                    padding="max_length",
                    return_tensors="pt",
                    truncation=True,
                ),
                key=batch_key,
            )
        case _:
            raise NotImplementedError(type(dc.tokenizer))


def add_text_transforms(dc: DatasetCollection, model_evaluator: ModelEvaluator) -> None:
    assert dc.dataset_type == DatasetType.Text
    assert has_torchtext
    dataset_name: str = dc.name.lower()
    # InputText
    if dataset_name == "imdb":
        dc.append_transform(
            functools.partial(replace_str, old="<br />", new=""),
            key=TransformType.InputText,
        )
        dc.append_transform(
            functools.partial(target_offset, offset=-1),
            key=TransformType.Target,
        )

    text_transforms = dc.dataset_kwargs.get("text_transforms", {})
    assert not text_transforms
    # for phase, transforms in text_transforms.items():
    #     for f in transforms:
    #         dc.append_transform(f, key=TransformType.InputText, phases=[phase])

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
    __apply_tokenizer_transforms(dc=dc, max_len=input_max_len, for_input=True)

    # Target
    match model_evaluator.model_type:
        case ModelType.TextGeneration:
            if dataset_name == "multi_nli":
                dc.append_transform(
                    functools.partial(
                        int_target_to_text,
                        mapping={0: "entailment", 1: "neutral", 2: "contradiction"},
                    ),
                    key=TransformType.Target,
                )
            elif isinstance(
                dc.get_dataset_util(
                    phase=MachineLearningPhase.Training
                ).get_sample_label(0),
                int,
            ):
                dc.append_transform(int_target_to_text, key=TransformType.Target)
            max_len = dc.dataset_kwargs.get("output_max_len", None)
            get_logger().info("use output text max len %s", max_len)
            __apply_tokenizer_transforms(dc=dc, max_len=max_len, for_input=False)
        case ModelType.Classification:
            assert isinstance(dc, ClassificationDatasetCollection)
            if isinstance(
                dc.get_dataset_util(
                    phase=MachineLearningPhase.Training
                ).get_sample_label(0),
                str,
            ):
                label_names = dc.get_label_names()
                dc.append_transform(
                    str_target_to_int(label_names), key=TransformType.Target
                )
