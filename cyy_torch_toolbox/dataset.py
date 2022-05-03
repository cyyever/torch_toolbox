import functools
from typing import Callable, Generator, Iterable

import torch
import torchvision


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
    dataset: torch.utils.data.IterableDataset,
) -> DictDataset:
    return DictDataset({idx: item for idx, item in enumerate(dataset)})


def sub_dataset(
    dataset: torch.utils.data.Dataset, indices: Iterable
) -> torch.utils.data.Dataset:
    r"""
    Subset of a dataset at specified indices in order.
    """
    indices = sorted(set(indices))
    if isinstance(dataset, torch.utils.data.IterableDataset):
        dataset = convert_iterable_dataset_to_map(dataset)
    return torch.utils.data.Subset(dataset, indices)


def sample_dataset(
    dataset: torch.utils.data.Dataset, index: int
) -> torch.utils.data.Dataset:
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


# import copy
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
