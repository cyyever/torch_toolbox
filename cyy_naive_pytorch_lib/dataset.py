import functools
import os
import random
from typing import Callable, Generator, Iterable

import PIL
import torch
import torchvision
from cyy_naive_lib.log import get_logger


class DatasetFilter:
    def __init__(self, dataset: torch.utils.data.Dataset, filters: Iterable[Callable]):
        self.__dataset = dataset
        self.__filters = filters
        self.__indices = None

    def __getitem__(self, index):
        return self.__dataset.__getitem__(self.indices[index])

    def __len__(self):
        return len(self.indices)

    @property
    def indices(self):
        if self.__indices is not None:
            return self.__indices
        indices = []
        for index, item in enumerate(self.__dataset):
            if all(f(index, item) for f in self.__filters):
                indices.append(index)
        self.__indices = indices
        return self.__indices


class DatasetMapper:
    def __init__(self, dataset: torch.utils.data.Dataset, mappers: Iterable[Callable]):
        self.__dataset = dataset
        self.__mappers = mappers

    def __getitem__(self, index):
        item = self.__dataset.__getitem__(index)
        for mapper in self.__mappers:
            item = mapper(index, item)
        return item

    def __len__(self):
        return len(self.__dataset)


def sub_dataset(dataset: torch.utils.data.Dataset, indices: Iterable):
    r"""
    Subset of a dataset at specified indices in order.
    """
    indices = sorted(set(indices))
    return torch.utils.data.Subset(dataset, indices)


def sample_dataset(dataset: torch.utils.data.Dataset, index: int):
    return sub_dataset(dataset, [index])


def dataset_with_indices(dataset: torch.utils.data.Dataset):
    def add_index(index, item):
        other_info = dict()
        feature = None
        target = None
        if len(item) == 3:
            feature, target, other_info = item
        else:
            feature, target = item
        other_info["index"] = index
        return (feature, target, other_info)

    return DatasetMapper(dataset, [add_index])


def split_dataset(dataset: torchvision.datasets.VisionDataset) -> Generator:
    return (torch.utils.data.Subset(dataset, [index]) for index in range(len(dataset)))


class DatasetUtil:
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset: torch.utils.data.Dataset = dataset
        self.__channel = None
        self.__len = None

    @property
    def len(self):
        if self.__len is None:
            self.__len = len(self.dataset)
        return self.__len

    @property
    def channel(self):
        if self.__channel is not None:
            return self.__channel
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1)
        channel = 0
        for x, _ in dataloader:
            channel = x.shape[1]
            break
        self.__channel = channel
        return self.__channel

    def get_mean_and_std(self):
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1)
        mean = torch.zeros(self.channel)
        for x, _ in dataloader:
            for i in range(self.channel):
                mean[i] += x[:, i, :, :].mean()
        mean.div_(self.len)

        wh = None
        std = torch.zeros(self.channel)
        for x, _ in dataloader:
            if wh is None:
                wh = x.shape[2] * x.shape[3]
            for i in range(self.channel):
                std[i] += torch.sum((x[:, i, :, :] - mean[i].data.item()) ** 2) / wh

        std = std.div(self.len).sqrt()
        return mean, std

    @staticmethod
    def get_labels_from_target(target) -> set:
        if isinstance(target, int):
            return set([target])
        if isinstance(target, torch.Tensor):
            return set(target.tolist())
        if isinstance(target, dict):
            if "labels" in target:
                return set(target["labels"].tolist())
        raise RuntimeError("can't extract labels from target: " + str(target))

    @staticmethod
    def get_label_from_target(target):
        labels = DatasetUtil.get_labels_from_target(target)
        assert len(labels) == 1
        return next(iter(labels))

    def get_sample_label(self, index):
        return DatasetUtil.get_label_from_target(self.dataset[index][1])

    def get_labels(self) -> set:
        def count_instance(container: set, instance):
            labels = DatasetUtil.get_labels_from_target(instance[1])
            container.update(labels)
            return container

        return functools.reduce(count_instance, self.dataset, set())

    def split_by_label(self) -> dict:
        label_map: dict = {}
        for index, _ in enumerate(self.dataset):
            label = self.get_sample_label(index)
            if label not in label_map:
                label_map[label] = []
            label_map[label].append(index)
        for label, indices in label_map.items():
            label_map[label] = dict()
            label_map[label]["indices"] = indices
        return label_map

    def get_label_number(self) -> int:
        return len(self.get_labels())

    def save_sample_image(self, idx, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if isinstance(self.dataset[idx][0], PIL.Image.Image):
            self.dataset[idx][0].save(path)
            return
        torchvision.utils.save_image(self.dataset[idx][0], path)

    def iid_split(self, parts: list) -> list:
        return self.__split(parts, by_label=True)

    def random_split(self, parts: list) -> list:
        return self.__split(parts, by_label=False)

    def __split(self, parts: list, by_label: bool = True) -> list:
        assert parts
        if len(parts) == 1:
            return [self.dataset]
        sub_dataset_indices_list: list = []
        for _ in parts:
            sub_dataset_indices_list.append([])

        if by_label:
            for _, v in self.split_by_label().items():
                label_indices_list = sorted(v["indices"])
                for i, part in enumerate(parts):
                    delimiter = int(len(label_indices_list) * part / sum(parts[i:]))
                    sub_dataset_indices_list[i] += label_indices_list[:delimiter]
                    label_indices_list = label_indices_list[delimiter:]
        else:
            label_indices_list = list(range(self.len))
            for i, part in enumerate(parts):
                delimiter = int(len(label_indices_list) * part / sum(parts[i:]))
                sub_dataset_indices_list[i] += label_indices_list[:delimiter]
                label_indices_list = label_indices_list[delimiter:]

        return [
            sub_dataset(self.dataset, indices) for indices in sub_dataset_indices_list
        ]

    def sample(self, percentage: float) -> Iterable:
        sample_size = int(self.len * percentage)
        return random.sample(range(self.len), k=sample_size)

    def iid_sample(self, percentage: float) -> dict:
        label_map = self.split_by_label()
        sample_indices = dict()
        for label, v in label_map.items():
            sample_size = int(len(v["indices"]) * percentage)
            if sample_size == 0:
                get_logger().warning("percentage is too small, use sample size 1")
                sample_size = 1
            sample_indices[label] = random.sample(v["indices"], k=sample_size)
        return sample_indices

    def randomize_subset_label(self, percentage: float) -> dict:
        sample_indices = self.iid_sample(percentage)
        labels = self.get_labels()
        randomized_label_map = dict()
        for label, indices in sample_indices.items():
            other_labels = list(set(labels) - set([label]))
            for index in indices:
                randomized_label_map[index] = random.choice(other_labels)
                assert randomized_label_map[index] != self.dataset[index][1]
        return randomized_label_map


def replace_dataset_labels(dataset, label_map: dict):
    assert label_map

    def mapper(index, item):
        if index in label_map:
            assert label_map[index] != item[1]
            item = list(item)
            item[1] = label_map[index]
            return tuple(item)
        return item

    return DatasetMapper(dataset, [mapper])
