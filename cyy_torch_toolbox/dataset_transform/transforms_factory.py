import functools
from typing import Any

import torch
from cyy_torch_toolbox.dependency import (has_hugging_face, has_torchtext,
                                          has_torchvision)

if has_torchtext:
    import torchtext

    from .tokenizer import SpacyTokenizer
    from .tokenizer_factory import get_spacy_tokenizer

if has_hugging_face:
    import transformers as hugging_face_transformers

    from .tokenizer_factory import get_hugging_face_tokenizer

if has_torchvision:
    import torchvision

from cyy_torch_toolbox.dataset_util import VisionDatasetUtil
from cyy_torch_toolbox.ml_type import (DatasetType, MachineLearningPhase,
                                       TransformType)

from .transforms import (Transforms, default_data_extraction,
                         str_target_to_int, swap_input_and_target)


def multi_nli_data_extraction(data: Any) -> dict:
    match data:
        case {"premise": premise, "hypothesis": hypothesis, "label": label, **kw}:
            item = {"input": [premise, hypothesis], "target": label}
            if "index" in kw:
                item["index"] = kw["index"]
            return item
        case _:
            return multi_nli_data_extraction(default_data_extraction(data))
    raise NotImplementedError()


def get_mean_and_std(dc):
    dataset = torch.utils.data.ConcatDataset(list(dc.foreach_raw_dataset()))
    transforms = Transforms()
    transforms.append(
        key=TransformType.Input, transform=torchvision.transforms.ToTensor()
    )

    def computation_fun():
        return VisionDatasetUtil(
            dataset=dataset,
            transforms=transforms,
            name=dc.name,
        ).get_mean_and_std()

    return dc.get_cached_data("mean_and_std.pk", computation_fun)


def create_multi_nli_text(sample_input, cls_token, sep_token):
    premise = sample_input[0]
    hypothesis = sample_input[1]
    return cls_token + " " + premise + " " + sep_token + " " + hypothesis


def replace_str(str, old, new):
    return str.replace(old, new)


def minus_one(target, index):
    return target - 1


def add_transforms(dc, dataset_kwargs=None, model_config=None):
    if dc.dataset_type == DatasetType.Vision:
        dc.append_transform(torchvision.transforms.ToTensor(), key=TransformType.Input)
        mean, std = get_mean_and_std(dc)
        dc.append_transform(
            torchvision.transforms.Normalize(mean=mean, std=std),
            key=TransformType.Input,
        )
        if dc.name.upper() not in ("SVHN", "MNIST"):
            dc.append_transform(
                torchvision.transforms.RandomHorizontalFlip(),
                key=TransformType.RandomInput,
                phases={MachineLearningPhase.Training},
            )
        if dc.name.upper() in ("CIFAR10", "CIFAR100"):
            dc.append_transform(
                torchvision.transforms.RandomCrop(32, padding=4),
                key=TransformType.RandomInput,
                phases={MachineLearningPhase.Training},
            )
        if dc.name.lower() == "imagenet":
            dc.append_transform(
                torchvision.transforms.RandomResizedCrop(224),
                key=TransformType.RandomInput,
                phases={MachineLearningPhase.Training},
            )
            dc.append_transform(
                torchvision.transforms.Resize(256),
                key=TransformType.Input,
                phases={MachineLearningPhase.Validation, MachineLearningPhase.Test},
            )
            dc.append_transform(
                torchvision.transforms.CenterCrop(224),
                key=TransformType.Input,
                phases={MachineLearningPhase.Validation, MachineLearningPhase.Test},
            )
        return
    if dc.dataset_type == DatasetType.Text and has_torchtext:
        # ExtractData
        dc.append_transform(swap_input_and_target, key=TransformType.ExtractData)
        if dc.name.lower() == "multi_nli":
            dc.clear_transform(key=TransformType.ExtractData)
            dc.append_transform(
                key=TransformType.ExtractData, transform=multi_nli_data_extraction
            )
        # InputText
        if dc.name.upper() == "IMDB":
            dc.append_transform(
                functools.partial(replace_str, old="<br />", new=""),
                key=TransformType.InputText,
            )
            dc.append_transform(
                minus_one,
                key=TransformType.Target,
            )

        if not dataset_kwargs:
            dataset_kwargs = {}

        text_transforms = dataset_kwargs.get("text_transforms", {})
        for phase, transforms in text_transforms.items():
            for f in transforms:
                dc.append_transform(f, key=TransformType.InputText, phases=[phase])

        # Input && InputBatch
        dc.tokenizer = None
        if model_config is not None:
            if "bert" in model_config.model_name:
                dc.tokenizer = get_hugging_face_tokenizer(
                    model_config.model_name.replace("sequence_classification_", "")
                )
        tokenizer_kwargs = dataset_kwargs.get("tokenizer", {})
        spacy_kwargs = tokenizer_kwargs.get("spacy_kwargs", {})
        create_spacy: bool = (
            dc.tokenizer is None
            or spacy_kwargs
            or tokenizer_kwargs.get("create_spacy", False)
        )
        if create_spacy:
            dc.spacy_tokenizer = get_spacy_tokenizer(dc, **spacy_kwargs)
        else:
            dc.spacy_tokenizer = None
        if dc.tokenizer is None:
            dc.tokenizer = dc.spacy_tokenizer

        max_len = dataset_kwargs.get("max_len", None)
        if max_len is None and model_config is not None:
            max_len = model_config.model_kwargs.get("max_len", None)
        match dc.tokenizer:
            case SpacyTokenizer():
                if dc.name.lower() == "multi_nli":
                    dc.append_transform(
                        functools.partial(
                            create_multi_nli_text,
                            cls_token="<cls>",
                            sep_token="<sep>",
                        ),
                        key=TransformType.InputText,
                    )
                dc.append_transform(dc.spacy_tokenizer, key=TransformType.Input)
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
            dc.get_dataset_util(phase=MachineLearningPhase.Training).get_sample_label(
                0
            ),
            str,
        ):
            label_names = dc.get_label_names()
            dc.append_transform(
                str_target_to_int(label_names), key=TransformType.Target
            )
