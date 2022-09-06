from typing import Callable, Generator, Iterable

import torch
import torchvision


def get_dataset_size(dataset: torch.utils.data.Dataset) -> int:
    try:
        return len(dataset)
    except BaseException:
        cnt: int = 0
        for _ in dataset:
            cnt += 1
        return cnt
    raise RuntimeError("not reachable")


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
        assert items
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
    assert indices
    match dataset:
        case DictDataset():
            return DictDataset(items=dict(enumerate([dataset[idx] for idx in indices])))
        case _:
            return torch.utils.data.Subset(dataset, indices)


def sample_dataset(
    dataset: torch.utils.data.Dataset, index: int
) -> torch.utils.data.Dataset:
    return sub_dataset(dataset, [index])


def add_index_to_item(index, item):
    return {"data": item, "index": index}


def dataset_with_indices(dataset: torch.utils.data.Dataset) -> DatasetMapper:
    return DatasetMapper(dataset, [add_index_to_item])


def split_dataset(dataset: torchvision.datasets.VisionDataset) -> Generator:
    dataset = convert_iterable_dataset_to_map(dataset)
    return (
        torch.utils.data.Subset(dataset, [index])
        for index in range(get_dataset_size(dataset))
    )
