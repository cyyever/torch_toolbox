from typing import Any, Generator, Iterable

import torch
import torchdata
import torchvision
from torchdata.datapipes.iter import IterableWrapper


def get_dataset_size(dataset: torch.utils.data.Dataset) -> int:
    try:
        return len(dataset)
    except BaseException:
        cnt: int = 0
        for _ in dataset:
            cnt += 1
        return cnt
    raise RuntimeError("not reachable")


class KeyPipe(torch.utils.data.MapDataPipe):
    def __init__(self, dp: torch.utils.data.MapDataPipe):
        super().__init__()
        self.__dp = dp

    def __getitem__(self, index):
        item = self.__dp.__getitem__(index)
        return (index, item)

    def __getattr__(self, attr):
        return getattr(self.__dp, attr)


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


def convert_item_to_data(item):
    return {"data": item}


def add_index_to_map_item(item):
    key, value = item[0], item[1]
    return convert_item_to_data(value) | {"index": key}


def dataset_with_indices(
    dataset: torch.utils.data.Dataset,
) -> torch.utils.data.MapDataPipe:
    if isinstance(dataset, torch.utils.data.IterableDataset):
        return IterableWrapper(dataset).map(convert_item_to_data).add_index()
    return torchdata.datapipes.map.Mapper(KeyPipe(dataset), add_index_to_map_item)


def get_iterable_item_key_and_value(item: Any) -> tuple:
    return item["index"], item


def convert_dataset_to_map_dp(
    dataset: torch.utils.data.IterableDataset,
) -> torch.utils.data.Dataset:
    dp = dataset_with_indices(dataset)
    if isinstance(dp, torch.utils.data.IterableDataset):
        return torchdata.datapipes.map.IterToMapConverter(
            dp, get_iterable_item_key_and_value
        )
    return dp


def sub_dataset(
    dataset: torch.utils.data.Dataset, indices: Iterable
) -> torch.utils.data.Dataset:
    r"""
    Subset of a dataset at specified indices in order.
    """
    assert indices
    indices = list(sorted(set(indices)))
    dataset = convert_dataset_to_map_dp(dataset)
    return torch.utils.data.Subset(dataset, indices)


def sample_dataset(
    dataset: torch.utils.data.Dataset, index: int
) -> torch.utils.data.Dataset:
    return sub_dataset(dataset, [index])


# def add_index_to_item(index, item):
#     return {"data": item, "index": index}


def split_dataset(dataset: torchvision.datasets.VisionDataset) -> Generator:
    dataset = convert_dataset_to_map_dp(dataset)
    return (
        torch.utils.data.Subset(dataset, [index])
        for index in range(get_dataset_size(dataset))
    )
