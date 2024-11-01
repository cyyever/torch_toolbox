from collections.abc import Generator
from typing import Any

import torch
import torch.utils.data
import torch.utils.data.datapipes
import torch.utils.data.dataset

from ..ml_type import OptionalIndicesType


def get_dataset_size(dataset: Any) -> int:
    match dataset:
        case {0: {"mask": mask}}:
            return mask.sum()
        case [{"mask": mask}]:
            return mask.sum()
    if hasattr(dataset, "__len__"):
        return len(dataset)
    match dataset:
        case torch.utils.data.IterableDataset():
            return sum(1 for _ in dataset)
    raise NotImplementedError(dataset)


class KeyPipe(torch.utils.data.MapDataPipe):
    def __init__(self, dp: Any) -> None:
        super().__init__()
        self.__dp = dp

    def __getitem__(self, index) -> tuple:
        item = self.__dp[index]
        return (index, item)

    def __len__(self) -> int:
        return len(self.__dp)


def __add_index_to_map_item(item) -> dict:
    key, value = item[0], item[1]
    return {"index": key, "data": value}


def dataset_with_indices(
    dataset: torch.utils.data.Dataset,
) -> torch.utils.data.Dataset:
    old_dataset = dataset
    match dataset:
        case list():
            return dataset
        case torch.utils.data.IterableDataset():
            dataset = torch.utils.data.datapipes.iter.IterableWrapper(dataset)
    match dataset:
        case torch.utils.data.IterDataPipe():
            dataset = dataset.enumerate()
        case _:
            dataset = torch.utils.data.datapipes.map.Mapper(
                KeyPipe(dataset), __add_index_to_map_item
            )
    assert not hasattr(dataset, "original_dataset")
    dataset.original_dataset = old_dataset
    return dataset


def select_item(dataset: Any, indices: OptionalIndicesType = None) -> Generator:
    if indices is not None:
        indices = set(indices)
    match dataset:
        case torch.utils.data.IterableDataset():
            iterator = iter(dataset)
            for idx, item in enumerate(iterator):
                if indices is None or idx in indices:
                    yield idx, item
                    if indices is not None:
                        indices.remove(idx)
        case _:
            if indices is None:
                indices = list(range(get_dataset_size(dataset)))
            for idx in indices:
                yield idx, dataset[idx]


def subset_dp(
    dataset: torch.utils.data.Dataset, indices: OptionalIndicesType = None
) -> torch.utils.data.MapDataPipe:
    return torch.utils.data.datapipes.map.SequenceWrapper(
        list(dict(select_item(dataset, indices)).values()), deepcopy=False
    )
