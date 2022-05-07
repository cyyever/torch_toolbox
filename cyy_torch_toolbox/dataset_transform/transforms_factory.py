import functools

import torch
import torchvision
from cyy_torch_toolbox.dataset_util import VisionDatasetUtil
from cyy_torch_toolbox.ml_type import (DatasetType, MachineLearningPhase,
                                       TransformType)

from .tokenizer import SpacyTokenizer
from .tokenizer_factory import get_tokenizer
from .transforms import Transforms, str_target_to_int, swap_input_and_target


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


def add_transforms(dc, dataset_kwargs):
    if dc.dataset_type == DatasetType.Vision:
        dc.append_transform(torchvision.transforms.ToTensor())
        mean, std = get_mean_and_std(dc)
        dc.append_transform(torchvision.transforms.Normalize(mean=mean, std=std))
        if dc.name.upper() not in ("SVHN", "MNIST"):
            dc.append_transform(
                torchvision.transforms.RandomHorizontalFlip(),
                phases={MachineLearningPhase.Training},
            )
        if dc.name.upper() in ("CIFAR10", "CIFAR100"):
            dc.append_transform(
                torchvision.transforms.RandomCrop(32, padding=4),
                phases={MachineLearningPhase.Training},
            )
        if dc.name.lower() == "imagenet":
            dc.append_transform(torchvision.transforms.RandomResizedCrop(224))
        return
    if dc.dataset_type == DatasetType.Text:
        # ExtractData
        dc.append_transform(swap_input_and_target, key=TransformType.ExtractData)
        if dc.name.lower() == "multi_nli":
            dc.clear_transform(key=TransformType.ExtractData)
            dc.append_transform(
                key=TransformType.ExtractData,
                transform=lambda data: (
                    (data["premise"], data["hypothesis"]),
                    data["label"],
                ),
            )
        # InputText
        if dc.name.upper() == "IMDB":
            dc.append_transform(
                lambda text: text.replace("<br />", ""), key=TransformType.InputText
            )

        text_transforms = dataset_kwargs.get("text_transforms", {})
        for phase, transforms in text_transforms.items():
            for f in transforms:
                dc.append_transform(f, key=TransformType.InputText, phases=[phase])

        # input
        dc.tokenizer = get_tokenizer(dataset_kwargs.get("tokenizer", {}), dc)
        if isinstance(dc.tokenizer, SpacyTokenizer):
            dc.append_transform(dc.tokenizer)
        else:
            dc.append_transform(dc.tokenizer)
        dc.append_transform(torch.LongTensor)
        # InputBatch
        if isinstance(dc.tokenizer, SpacyTokenizer):
            dc.append_transform(
                functools.partial(
                    torch.nn.utils.rnn.pad_sequence,
                    padding_value=dc.tokenizer.vocab["<pad>"],
                ),
                key=TransformType.InputBatch,
            )
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
