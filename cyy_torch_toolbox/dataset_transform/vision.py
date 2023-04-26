import torch

from ..dataset_util import VisionDatasetUtil
from ..dependency import has_torchvision
from ..ml_type import DatasetType, MachineLearningPhase, TransformType
from .transforms import Transforms

if has_torchvision:
    import torchvision


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


def add_vision_transforms(dc, dataset_kwargs=None, model_config=None):
    assert dc.dataset_type == DatasetType.Vision
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
