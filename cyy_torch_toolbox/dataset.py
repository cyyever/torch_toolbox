# import copy
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
        self.__mappers = list(mappers)

    @property
    def dataset(self):
        return self.__dataset

    def __getitem__(self, index):
        item = self.__dataset.__getitem__(index)
        for mapper in self.__mappers:
            item = mapper(index, item)
        return item

    def add_mapper(self, mapper: Callable) -> None:
        self.__mappers.append(mapper)

    def __len__(self):
        return len(self.__dataset)


class DictDataset(torch.utils.data.Dataset):
    def __init__(self, items: dict):
        super().__init__()
        self.__items = items

    def __getitem__(self, index):
        if index not in self.__items:
            raise StopIteration()
        return self.__items[index]

    def __len__(self):
        return len(self.__items)


def convert_iterable_dataset_to_map(
    dataset: torch.utils.data.IterableDataset, swap_item: bool = False
) -> dict:
    items = {}
    for idx, item in enumerate(dataset):
        if swap_item:
            items[idx] = (item[1], item[0])
        else:
            items[idx] = item
    return DictDataset(items)


def sub_dataset(dataset: torch.utils.data.Dataset, indices: Iterable):
    r"""
    Subset of a dataset at specified indices in order.
    """
    indices = sorted(set(indices))
    subset = torch.utils.data.Subset(dataset, indices)
    if hasattr(dataset, "sort_key"):
        setattr(subset, "sort_key", dataset.sort_key)
    return subset


def sample_dataset(dataset: torch.utils.data.Dataset, index: int):
    return sub_dataset(dataset, [index])


def __add_index_to_item(index, item):
    other_info = {}
    feature = None
    target = None
    if len(item) == 3:
        feature, target, other_info = item
    else:
        feature, target = item
    other_info["index"] = index
    return (feature, target, other_info)


def dataset_with_indices(dataset: torch.utils.data.Dataset):
    return DatasetMapper(dataset, [__add_index_to_item])


def split_dataset(dataset: torchvision.datasets.VisionDataset) -> Generator:
    return (torch.utils.data.Subset(dataset, [index]) for index in range(len(dataset)))


class DatasetUtil:
    def __init__(
        self, dataset: torch.utils.data.Dataset, transforms=None, target_transforms=None
    ):
        self.dataset: torch.utils.data.Dataset = dataset
        self.__channel = None
        self.__len = None
        if transforms is None:
            transforms = []
        self.__transforms = transforms

        if target_transforms is None:
            target_transforms = []
        self.__target_transforms = target_transforms

    # def get_transforms(self):
    #     transforms = []
    #     if hasattr(self.dataset, "transform"):
    #         transforms = [self.dataset.transform]
    #     i = 0
    #     while i < len(transforms):
    #         if isinstance(transforms[i], torchvision.transforms.Compose):
    #             transforms = (
    #                 transforms[:i] + transforms[i].transforms + transforms[i + 1:]
    #             )
    #         else:
    #             i += 1
    #     return transforms

    @property
    def len(self):
        if self.__len is None:
            self.__len = len(self.dataset)
        return self.__len

    @property
    def channel(self):
        if isinstance(self.dataset, torch.utils.data.IterableDataset):
            raise RuntimeError("IterableDataset is not supported")
        if self.__channel is not None:
            return self.__channel
        x = next(iter(self.dataset))[0]
        self.__channel = x.shape[0]
        assert self.__channel <= 3
        return self.__channel

    def get_mean_and_std(self):
        mean = torch.zeros(self.channel)
        for idx in range(len(self.dataset)):
            x = self.dataset[idx][0]
            for i in range(self.channel):
                mean[i] += x[i, :, :].mean()
        mean.div_(self.len)

        wh = None
        std = torch.zeros(self.channel)
        for idx in range(len(self.dataset)):
            x = self.dataset[idx][0]
            if wh is None:
                wh = x.shape[1] * x.shape[2]
            for i in range(self.channel):
                std[i] += torch.sum((x[i, :, :] - mean[i].data.item()) ** 2) / wh
        std = std.div(self.len).sqrt()
        return mean, std

    @classmethod
    def __decode_target(cls, target) -> set:
        if isinstance(target, int):
            return set([target])
        if isinstance(target, list):
            return set(target)
        if isinstance(target, torch.Tensor):
            return cls.__decode_target(target.tolist())
        if isinstance(target, str):
            if target == "neg":
                return set([0])
            if target == "pos":
                return set([1])
        if isinstance(target, dict):
            if "labels" in target:
                return set(target["labels"].tolist())
            if all(isinstance(s, str) and s.isnumeric() for s in target):
                return set(int(s) for s in target)

        # match target:
        #     case int():
        #         return set([target])
        #     case list():
        #         return set(target)
        #     case torch.Tensor():
        #         return DatasetUtil.__decode_target(target.tolist())
        #     case str():
        #         if target == "neg":
        #             return set([0])
        #         if target == "pos":
        #             return set([1])
        #     case dict():
        #         if "labels" in target:
        #             return set(target["labels"].tolist())
        raise RuntimeError("can't extract labels from target: " + str(target))

    def get_sample_labels(self, index, dataset=None):
        if dataset is None:
            dataset = self.dataset
        if isinstance(dataset, torch.utils.data.Subset):
            return self.get_sample_label(dataset.indices[index], dataset.dataset)
        if hasattr(dataset, "targets"):
            target = dataset.targets[index]
        else:
            target = dataset[index][1]
        for f in self.__target_transforms:
            target = f(target)
        return DatasetUtil.__decode_target(target)

    def get_sample_label(self, index):
        labels = self.get_sample_labels(index)
        assert len(labels) == 1
        return next(iter(labels))

    def get_labels(self) -> set:
        labels = set()
        for i in range(len(self.dataset)):
            labels.update(self.get_sample_labels(index=i))
        return labels

    def get_label_names(self) -> dict:
        if hasattr(self.dataset, "classes"):
            classes = getattr(self.dataset, "classes")
            if classes and isinstance(classes[0], str):
                return dict(enumerate(classes))

        def get_label_name(container: set, instance) -> set:
            label = instance[1]
            if isinstance(label, str):
                container.add(label)
            return container

        label_names = functools.reduce(get_label_name, self.dataset, set())
        if label_names:
            return dict(enumerate(sorted(label_names)))
        return None

    def split_by_label(self) -> dict:
        label_map: dict = {}
        for index in range(len(self.dataset)):
            label = self.get_sample_label(index)
            if label not in label_map:
                label_map[label] = []
            label_map[label].append(index)
        for label, indices in label_map.items():
            label_map[label] = {}
            label_map[label]["indices"] = indices
        return label_map

    def get_label_number(self) -> int:
        return len(self.get_labels())

    def save_sample_image(self, idx: int, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if isinstance(self.dataset[idx][0], PIL.Image.Image):
            self.dataset[idx][0].save(path)
            return
        torchvision.utils.save_image(self.dataset[idx][0], path)

    @torch.no_grad()
    def get_sample_image(self, idx: int) -> PIL.Image:
        tensor = self.dataset[idx][0]
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

    def get_sample_text(self, idx: int) -> str:
        return self.dataset[idx][0]

    def iid_split_indices(self, parts: list) -> list:
        return self.__get_split_indices(parts, by_label=True)

    def random_split_indices(self, parts: list) -> list:
        return self.__get_split_indices(parts, by_label=False)

    def iid_split(self, parts: list) -> list:
        return self.__split(parts, by_label=True)

    def random_split(self, parts: list) -> list:
        return self.__split(parts, by_label=False)

    def split_by_indices(self, indices_list: list) -> list:
        return [sub_dataset(self.dataset, indices) for indices in indices_list]

    def __get_split_indices(self, parts: list, by_label: bool = True) -> list:
        assert parts
        sub_dataset_indices_list: list = []
        if len(parts) == 1:
            sub_dataset_indices_list.append(list(range(len(self.dataset))))
            return sub_dataset_indices_list
        for _ in parts:
            sub_dataset_indices_list.append([])

        if by_label:
            for v in self.split_by_label().values():
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
        return sub_dataset_indices_list

    def __split(self, parts: list, by_label: bool = True) -> list:
        assert parts
        if len(parts) == 1:
            return [self.dataset]
        sub_dataset_indices_list = self.__get_split_indices(parts, by_label)
        return self.split_by_indices(sub_dataset_indices_list)

    def sample(self, percentage: float) -> Iterable:
        sample_size = int(self.len * percentage)
        return random.sample(range(self.len), k=sample_size)

    def iid_sample(self, percentage: float) -> dict:
        label_map = self.split_by_label()
        sample_indices = {}
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
        randomized_label_map = {}
        for label, indices in sample_indices.items():
            other_labels = list(set(labels) - set([label]))
            for index in indices:
                randomized_label_map[index] = random.choice(other_labels)
                assert randomized_label_map[index] != self.dataset[index][1]
        return randomized_label_map


# class CachedVisionDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset: torch.utils.data.Dataset):
#         self.__dataset = copy.deepcopy(dataset)
#         transforms = DatasetUtil(self.__dataset).get_transforms()
#         remain_transforms = []
#         while transforms:
#             t = transforms[0]
#             transforms.pop(0)
#             if isinstance(t, torchvision.transforms.ToTensor):
#                 remain_transforms.append(t)
#         self.__dataset.transform = torchvision.transforms.Compose(remain_transforms)
#         self.transform = torchvision.transforms.Compose(transforms)

#         self.__items: dict = {}

#     def __getitem__(self, index):
#         if index in self.__items:
#             item = self.__items[index]
#         else:
#             item = self.__dataset[index]
#             self.__items[index] = item
#         img, target = item
#         return self.transform(img), target

#     def __len__(self):
#         return len(self.__dataset)


def replace_dataset_labels(dataset, label_map: dict):
    assert label_map

    def __replace_item_label(label_map, index, item):
        if index in label_map:
            assert label_map[index] != item[1]
            item = list(item)
            item[1] = label_map[index]
            return tuple(item)
        return item

    return DatasetMapper(dataset, [functools.partial(__replace_item_label, label_map)])


def decode_batch(batch):
    if hasattr(batch, "text"):
        return (getattr(batch, "text"), getattr(batch, "label"), {})
    if len(batch) == 1:
        batch = batch[0]
        assert isinstance(batch, dict)
        sample_inputs = batch["data"]
        sample_targets = batch["label"].squeeze(-1).long()
    else:
        sample_inputs = batch[0]
        sample_targets = batch[1]
        if len(batch) == 3:
            return (sample_inputs, sample_targets, batch[2])
    return (sample_inputs, sample_targets, {})
