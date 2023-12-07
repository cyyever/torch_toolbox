import torch
import torch.utils.data
import torchvision
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.dataset.collection import DatasetCollection
from cyy_torch_toolbox.ml_type import (DatasetType, MachineLearningPhase,
                                       TransformType)

from ..dataset.vision_util import VisionDatasetUtil
from .transform import Transforms


def get_mean_and_std(dc):
    dataset = torch.utils.data.ConcatDataset(list(dc.foreach_dataset()))
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


def add_vision_extraction(dc: DatasetCollection) -> None:
    assert dc.dataset_type == DatasetType.Vision
    dc.append_transform(torchvision.transforms.ToTensor(), key=TransformType.Input)


def add_vision_transforms(dc: DatasetCollection, model_evaluator) -> None:
    assert dc.dataset_type == DatasetType.Vision
    mean, std = get_mean_and_std(dc)
    dc.append_transform(
        torchvision.transforms.Normalize(mean=mean, std=std),
        key=TransformType.Input,
    )
    input_size = getattr(
        model_evaluator.get_underlying_model().__class__, "input_size", None
    )
    if input_size is not None:
        get_logger().debug("resize input to %s", input_size)
        dc.append_transform(
            transform=torchvision.transforms.Resize(input_size, antialias=True),
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
