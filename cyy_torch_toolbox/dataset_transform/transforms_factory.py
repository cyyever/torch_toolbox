import functools
from typing import Any

import torch
import torchtext
import torchvision
import transformers
from cyy_torch_toolbox.dataset_util import VisionDatasetUtil
from cyy_torch_toolbox.ml_type import (DatasetType, MachineLearningPhase,
                                       TransformType)

from .tokenizer import SpacyTokenizer
from .tokenizer_factory import get_tokenizer
from .transforms import Transforms, str_target_to_int, swap_input_and_target


def multi_nli_data_extraction(data: Any) -> dict:
    match data:
        case {"data": real_data, "index": index}:
            return multi_nli_data_extraction(real_data) | {"index": index}
        case {"premise": premise, "hypothesis": hypothesis, "label": label, **kw}:
            return {"input": [premise, hypothesis], "target": label}
    raise NotImplementedError()


def get_mean_and_std(dc):
    dataset = torch.utils.data.ConcatDataset(list(dc._datasets.values()))
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

    return dc._get_cache_data("mean_and_std.pk", computation_fun)


def replace_str(str, old, new):
    return str.replace(old, new)


def add_transforms(dc, dataset_kwargs, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {}
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
        return
    if dc.dataset_type == DatasetType.Text:
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

        text_transforms = dataset_kwargs.get("text_transforms", {})
        for phase, transforms in text_transforms.items():
            for f in transforms:
                dc.append_transform(f, key=TransformType.InputText, phases=[phase])

        # Input && InputBatch
        dc.tokenizer = get_tokenizer(dataset_kwargs.get("tokenizer", {}), dc)
        max_len = model_kwargs.get("max_len", None)
        if max_len is None:
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
            case transformers.PreTrainedTokenizerBase():
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
                raise NotImplementedError()
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