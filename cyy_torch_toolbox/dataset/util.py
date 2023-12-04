import functools
import os
from collections.abc import Iterable
from typing import Any, Generator, Type

import torch
import torch.utils.data
import torchvision

from ..data_pipeline.transform import Transforms
from ..factory import Factory
from ..ml_type import DatasetType
from . import get_dataset_size, select_item, subset_dp


class DatasetUtil:
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        name: None | str = None,
        transforms: Transforms | None = None,
        cache_dir: None | str = None,
    ) -> None:
        self.dataset: torch.utils.data.Dataset | list = dataset
        self.__len: None | int = None
        self._name: str = name if name else ""
        self.__transforms: Transforms | None = transforms
        self._cache_dir = cache_dir

    def __len__(self) -> int:
        if self.__len is None:
            self.__len = get_dataset_size(self.dataset)
        return self.__len

    def decompose(self) -> None | dict:
        return None

    def get_subset(self, indices: Iterable) -> list[dict]:
        return subset_dp(self.dataset, indices)

    def get_samples(self, indices: Iterable | None = None) -> Iterable:
        items = select_item(dataset=self.dataset, indices=indices, mask=self.get_mask())
        if self.__transforms is None:
            return items
        for idx, sample in items:
            sample = self.__transforms.extract_data(sample)
            yield idx, sample

    def get_mask(self) -> None | list[torch.Tensor]:
        return None

    def get_sample(self, index: int) -> Any:
        for _, sample in self.get_samples(indices=[index]):
            return sample
        return None

    @classmethod
    def __decode_target(cls: Type, target: Any) -> set:
        match target:
            case int() | str():
                return {target}
            case list():
                return set(target)
            case torch.Tensor():
                if target.numel() == 1:
                    return {target.item()}
                if (target <= 1).all().item() and (target >= 0).all().item():
                    # one hot vector
                    return set(target.nonzero().view(-1).tolist())
                raise NotImplementedError(f"Unsupported target {target}")
            case dict():
                if "labels" in target:
                    return set(target["labels"].tolist())
                if all(isinstance(s, str) and s.isnumeric() for s in target):
                    return {int(s) for s in target}
        raise RuntimeError("can't extract labels from target: " + str(target))

    def _get_sample_input(self, index: int, apply_transform: bool = True) -> Any:
        sample = self.get_sample(index)
        sample_input = sample["input"]
        if apply_transform:
            assert self.__transforms is not None
            sample_input = self.__transforms.transform_input(
                sample_input, apply_random=False
            )
        return sample_input

    def get_batch_labels(self, indices: None | Iterable = None) -> Generator:
        for idx, sample in self.get_samples(indices):
            target = sample["target"]
            if self.__transforms is not None:
                target = self.__transforms.transform_target(target)
            yield idx, DatasetUtil.__decode_target(target)

    def get_sample_label(self, index):
        for _, labels in self.get_batch_labels(indices=[index]):
            assert len(labels) == 1
            return next(iter(labels))
        return None

    def get_labels(self) -> set:
        return set().union(*tuple(set(labels) for _, labels in self.get_batch_labels()))

    def get_original_dataset(self) -> torch.utils.data.Dataset:
        return self.dataset[0].get("original_dataset", self.dataset)

    def get_label_names(self) -> dict:
        original_dataset = self.get_original_dataset()
        if (
            hasattr(original_dataset, "classes")
            and original_dataset.classes
            and isinstance(original_dataset.classes[0], str)
        ):
            return dict(enumerate(original_dataset.classes))

        def get_label_name(container: set, index: int) -> set:
            label = self.get_sample_label(index)
            if isinstance(label, str):
                container.add(label)
            return container

        label_names: set = functools.reduce(get_label_name, range(len(self)), set())
        if label_names:
            return dict(enumerate(sorted(label_names)))
        raise RuntimeError("no label names detected")


class VisionDatasetUtil(DatasetUtil):
    @functools.cached_property
    def channel(self):
        x = self._get_sample_input(0)
        assert x.shape[0] <= 3
        return x.shape[0]

    def get_mean_and_std(self):
        if self._name.lower() == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            return (mean, std)
        mean = torch.zeros(self.channel)
        for index in range(len(self)):
            x = self._get_sample_input(index)
            for i in range(self.channel):
                mean[i] += x[i, :, :].mean()
        mean.div_(len(self))

        wh = None
        std = torch.zeros(self.channel)
        for index in range(len(self)):
            x = self._get_sample_input(index)
            if wh is None:
                wh = x.shape[1] * x.shape[2]
            for i in range(self.channel):
                std[i] += torch.sum((x[i, :, :] - mean[i].data.item()) ** 2) / wh
        std = std.div(len(self)).sqrt()
        return mean, std

    def save_sample_image(self, index: int, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sample_input = self._get_sample_input(index, apply_transform=False)
        if "image" in sample_input.__class__.__name__.lower():
            return sample_input
        torchvision.utils.save_image(sample_input, path)


global_dataset_util_factor = Factory()
global_dataset_util_factor.register(DatasetType.Vision, VisionDatasetUtil)
