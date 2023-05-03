import functools
from typing import Any

import torch

from ..dataset_collection import DatasetCollection
from ..dependency import has_hugging_face, has_spacy, has_torchtext
from ..ml_type import DatasetType, MachineLearningPhase, TransformType
from ..model_evaluator import ModelEvaluator
from .common import (default_data_extraction, replace_str, str_target_to_int,
                     swap_input_and_target, target_offset)
from .template import get_text_template, interpret_template

if has_torchtext:
    import torchtext
if has_spacy:
    from cyy_torch_toolbox.tokenizer.spacy import SpacyTokenizer

if has_hugging_face:
    import transformers as hugging_face_transformers


def multi_nli_data_extraction(data: Any) -> dict:
    match data:
        case {"premise": premise, "hypothesis": hypothesis, "label": label, **kwargs}:
            item = {
                "input": {"premise": premise, "hypothesis": hypothesis},
                "target": label,
            }
            if "index" in kwargs:
                item["index"] = kwargs["index"]
            return item
        case _:
            return multi_nli_data_extraction(default_data_extraction(data))
    raise NotImplementedError()


def create_multi_nli_text(sample_input, cls_token, sep_token):
    premise = sample_input[0]
    hypothesis = sample_input[1]
    return cls_token + " " + premise + " " + sep_token + " " + hypothesis


def add_text_extraction(dc: DatasetCollection) -> None:
    assert dc.dataset_type == DatasetType.Text
    assert has_torchtext
    # ExtractData
    if dc.name is not None:
        match dc.name.lower():
            # case "multi_nli":
            #     dc.clear_transform(key=TransformType.ExtractData)
            #     dc.append_transform(
            #         key=TransformType.ExtractData, transform=multi_nli_data_extraction
            #     )
            case "imdb":
                dc.append_transform(
                    swap_input_and_target, key=TransformType.ExtractData
                )


def add_text_transforms(
    dc: DatasetCollection, model_evaluator: ModelEvaluator, dataset_kwargs: dict
) -> None:
    assert dc.dataset_type == DatasetType.Text
    assert has_torchtext
    if not dataset_kwargs:
        dataset_kwargs = {}
    dataset_name: str = ""
    if dc.name is not None:
        dataset_name = dc.name.lower()
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

    # text_transforms = dataset_kwargs.get("text_transforms", {})
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
    max_len = dataset_kwargs.get("max_len", None)
    match dc.tokenizer:
        case SpacyTokenizer():
            dc.append_transform(dc.tokenizer, key=TransformType.Input)
            if max_len is not None:
                dc.append_transform(
                    torchtext.transforms.Truncate(max_seq_len=max_len),
                    key=TransformType.Input,
                )
            dc.append_transform(torch.LongTensor, key=TransformType.Input)
            dc.append_transform(
                functools.partial(
                    torch.nn.utils.rnn.pad_sequence,
                    padding_value=dc.tokenizer.vocab["<pad>"],
                ),
                key=TransformType.InputBatch,
            )
        case hugging_face_transformers.PreTrainedTokenizerBase():
            assert max_len is not None
            dc.append_transform(
                functools.partial(
                    dc.tokenizer,
                    max_length=max_len,
                    padding="max_length",
                    return_tensors="pt",
                    truncation=True,
                ),
                key=TransformType.InputBatch,
            )
        case _:
            raise NotImplementedError(str(type(dc.tokenizer)))
    # Target
    if isinstance(
        dc.get_dataset_util(phase=MachineLearningPhase.Training).get_sample_label(0),
        str,
    ):
        label_names = dc.get_label_names()
        dc.append_transform(str_target_to_int(label_names), key=TransformType.Target)
