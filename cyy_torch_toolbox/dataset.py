from typing import Callable, Generator, Iterable

import torch
import torchvision


def get_dataset_size(dataset) -> int:
    try:
        return len(dataset)
    except BaseException:
        cnt: int = 0
        for _ in dataset:
            cnt += 1
        return cnt
    raise RuntimeError("not reachable")


class DatasetFilter:
    def __init__(
        self, dataset: torch.utils.data.MapDataPipe, filters: Iterable[Callable]
    ):
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
    def __init__(
        self, dataset: torch.utils.data.MapDataPipe, mappers: Iterable[Callable]
    ):
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


class DictDataset(torch.utils.data.MapDataPipe):
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
    if isinstance(dataset, torch.utils.data.IterableDataset):
        return DictDataset(dict(enumerate(dataset)))
    return dataset


def sub_dataset(
    dataset: torch.utils.data.Dataset, indices: Iterable
) -> torch.utils.data.Dataset:
    r"""
    Subset of a dataset at specified indices in order.
    """
    indices = sorted(set(indices))
    dataset = convert_iterable_dataset_to_map(dataset)
    return torch.utils.data.Subset(dataset, indices)


def sample_dataset(
    dataset: torch.utils.data.Dataset, index: int
) -> torch.utils.data.Dataset:
    return sub_dataset(dataset, [index])


def __add_index_to_item(index, item):
    return {"data": item, "index": index}


def dataset_with_indices(dataset: torch.utils.data.Dataset):
    return DatasetMapper(dataset, [__add_index_to_item])


def split_dataset(dataset: torchvision.datasets.VisionDataset) -> Generator:
    dataset = convert_iterable_dataset_to_map(dataset)
    return (
        torch.utils.data.Subset(dataset, [index])
        for index in range(get_dataset_size(dataset))
    )
