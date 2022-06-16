import functools
import os
import random
from typing import Iterable

import PIL
import torch
import torchvision
from cyy_naive_lib.log import get_logger

from .dataset import get_dataset_size, sub_dataset
from .dataset_transform.transforms import Transforms


class DatasetUtil:
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        transforms: Transforms = None,
        name=None,
    ):
        self.dataset: torch.utils.data.Dataset = dataset
        self.__len = None
        self._name: str | None = name
        self.__transforms = transforms

    def __len__(self):
        if self.__len is None:
            self.__len = get_dataset_size(self.dataset)
        return self.__len

    def get_sample(self, index: int):
        if isinstance(self.dataset, torch.utils.data.IterableDataset):
            if hasattr(self.dataset, "reset"):
                self.dataset.reset()
            iterator = iter(self.dataset)
            for _ in range(index):
                next(iterator)
            sample = next(iterator)
            if hasattr(self.dataset, "reset"):
                self.dataset.reset()
        else:
            sample = self.dataset[index]
        if self.__transforms is not None:
            sample = self.__transforms.extract_data(sample)
        return sample

    @classmethod
    def __decode_target(cls, target) -> set:
        match target:
            case int() | str():
                return set([target])
            case list():
                return set(target)
            case torch.Tensor():
                return cls.__decode_target(target.tolist())
            case dict():
                if "labels" in target:
                    return set(target["labels"].tolist())
                if all(isinstance(s, str) and s.isnumeric() for s in target):
                    return set(int(s) for s in target)
        raise RuntimeError("can't extract labels from target: " + str(target))

    def _get_sample_input(self, index: int, apply_transform: bool = True):
        sample = self.get_sample(index)
        sample_input = sample["input"]
        if apply_transform:
            assert self.__transforms is not None
            sample_input = self.__transforms.transform_input(
                sample_input, apply_random=False
            )
        return sample_input

    def get_sample_labels(self, index: int) -> set:
        if isinstance(self.dataset, torch.utils.data.Subset):
            dataset = self.dataset.dataset
            new_index = self.dataset.indices[index]
        else:
            dataset = self.dataset
            new_index = index
        if hasattr(dataset, "targets") and len(dataset.targets) >= len(self):
            target = dataset.targets[new_index]
        else:
            target = self.get_sample(index)["target"]
        if self.__transforms is not None:
            target = self.__transforms.transform_target(target)
        return DatasetUtil.__decode_target(target)

    def get_sample_label(self, index):
        labels = self.get_sample_labels(index)
        assert len(labels) == 1
        return next(iter(labels))

    def get_labels(self) -> set:
        return set().union(
            *tuple(self.get_sample_labels(index=i) for i in range(len(self)))
        )

    def get_label_names(self) -> dict:
        if hasattr(self.dataset, "classes"):
            classes = getattr(self.dataset, "classes")
            if classes and isinstance(classes[0], str):
                return dict(enumerate(classes))

        def get_label_name(container: set, idx) -> set:
            label = self.get_sample_label(idx)
            if isinstance(label, str):
                container.add(label)
            return container

        label_names = functools.reduce(get_label_name, range(len(self)), set())
        if label_names:
            return dict(enumerate(sorted(label_names)))
        raise RuntimeError("not label names detected")


class DatasetSplitter(DatasetUtil):
    __sample_label_dict = None
    __label_sample_dict = None

    @property
    def sample_label_dict(self) -> dict[int, list]:
        if self.__sample_label_dict is not None:
            return self.__sample_label_dict
        self.__sample_label_dict = {}
        for index in range(len(self)):
            self.__sample_label_dict[index] = list(self.get_sample_labels(index))
        return self.__sample_label_dict

    @property
    def label_sample_dict(self) -> dict:
        if self.__label_sample_dict is not None:
            return self.__label_sample_dict
        self.__label_sample_dict = {}
        for index, labels in self.sample_label_dict.items():
            for label in labels:
                if label not in self.__label_sample_dict:
                    self.__label_sample_dict[label] = [index]
                else:
                    self.__label_sample_dict[label].append(index)
        return self.__label_sample_dict

    def get_label_number(self) -> int:
        return len(self.get_labels())

    def get_sample_text(self, idx: int) -> str:
        return self.dataset[idx][0]

    def iid_split_indices(self, parts: list) -> list:
        return self.__get_split_indices(parts, iid=True)

    def random_split_indices(self, parts: list) -> list:
        return self.__get_split_indices(parts, iid=False)

    def iid_split(self, parts: list) -> list:
        return self.__split(parts, iid=True)

    def split_by_indices(self, indices_list: list) -> list:
        return [sub_dataset(self.dataset, indices) for indices in indices_list]

    def __get_split_indices(self, parts: list, iid: bool = True) -> list[list]:
        assert parts
        if len(parts) == 1:
            return [list(range(len(self)))]

        def split_idx_impl(indices_list: list) -> list[list]:
            part_lens = []
            for part in parts:
                part_len = int(len(indices_list) * part / sum(parts))
                assert part_len != 0
                part_lens.append(part_len)
            part_lens[-1] += len(indices_list) - sum(part_lens)
            part_indices = []
            for part_len in part_lens:
                part_indices.append(indices_list[0:part_len])
                indices_list = indices_list[part_len:]
            return part_indices

        if not iid:
            index_list = list(range(len(self)))
            random.shuffle(index_list)
            return split_idx_impl(index_list)

        sub_index_list: list[list] = []
        for _ in parts:
            sub_index_list.append([])
        for v in self.label_sample_dict.values():
            part_index_list = split_idx_impl(sorted(v))
            random.shuffle(part_index_list)
            for a, b in zip(sub_index_list, part_index_list):
                a += b
        return sub_index_list

    def __split(self, parts: list, iid: bool = True) -> list:
        assert parts
        if len(parts) == 1:
            return [self.dataset]
        sub_dataset_indices_list = self.__get_split_indices(parts, iid)
        return self.split_by_indices(sub_dataset_indices_list)

    def sample(self, percentage: float) -> Iterable:
        sample_size = int(len(self) * percentage)
        return random.sample(range(len(self)), k=sample_size)

    def iid_sample(self, percentage: float) -> dict:
        sample_indices = {}
        for label, v in self.label_sample_dict.items():
            sample_size = int(len(v) * percentage)
            if sample_size == 0:
                get_logger().warning("percentage is too small, use sample size 1")
                sample_size = 1
            sample_indices[label] = random.sample(v, k=sample_size)
        return sample_indices

    def randomize_subset_label(self, percentage: float) -> dict:
        sample_indices = self.iid_sample(percentage)
        labels = self.get_labels()
        randomized_label_map = {}
        for label, indices in sample_indices.items():
            other_labels = list(set(labels) - set([label]))
            for index in indices:
                randomized_label_map[index] = random.choice(other_labels)
        return randomized_label_map


class VisionDatasetUtil(DatasetSplitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__channel = None

    @property
    def channel(self):
        if self.__channel is not None:
            return self.__channel
        x = self._get_sample_input(0)
        self.__channel = x.shape[0]
        assert self.__channel <= 3
        return self.__channel

    def get_mean_and_std(self):
        if self._name is not None and self._name.lower() == "imagenet":
            mean = torch.Tensor([0.485, 0.456, 0.406])
            std = torch.Tensor([0.229, 0.224, 0.225])
            return (mean, std)
        mean = torch.zeros(self.channel)
        for idx in range(len(self)):
            x = self._get_sample_input(idx)
            for i in range(self.channel):
                mean[i] += x[i, :, :].mean()
        mean.div_(len(self))

        wh = None
        std = torch.zeros(self.channel)
        for idx in range(len(self)):
            x = self._get_sample_input(idx)
            if wh is None:
                wh = x.shape[1] * x.shape[2]
            for i in range(self.channel):
                std[i] += torch.sum((x[i, :, :] - mean[i].data.item()) ** 2) / wh
        std = std.div(len(self)).sqrt()
        return mean, std

    def save_sample_image(self, idx: int, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sample_input = self._get_sample_input(idx, apply_transform=False)
        match sample_input:
            case PIL.Image.Image():
                sample_input.save(path)
            case _:
                torchvision.utils.save_image(sample_input, path)

    @torch.no_grad()
    def get_sample_image(self, idx: int) -> PIL.Image:
        tensor = self._get_sample_input(idx, apply_transform=False)
        if isinstance(tensor, PIL.Image.Image):
            return tensor
        grid = torchvision.utils.make_grid(tensor)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = (
            grid.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )
        return PIL.Image.fromarray(ndarr)


class TextDatasetUtil(DatasetSplitter):
    @torch.no_grad()
    def get_sample_text(self, idx: int) -> str:
        return self._get_sample_input(idx, apply_transform=False)
