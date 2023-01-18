from typing import Any, Generator

import torch
import torchdata


def get_dataset_size(dataset: torch.utils.data.Dataset) -> int:
    if isinstance(dataset, torch.utils.data.IterableDataset):
        cnt: int = 0
        for _ in dataset:
            cnt += 1
        return cnt
    _ = next(iter(dataset))
    return len(dataset)


class KeyPipe(torch.utils.data.MapDataPipe):
    def __init__(self, dp: torch.utils.data.MapDataPipe):
        super().__init__()
        self.__dp = dp

    def __getitem__(self, index) -> tuple:
        item = self.__dp.__getitem__(index)
        return (index, item)

    def __len__(self) -> int:
        return len(self.__dp)


def __convert_item_to_dict(item) -> dict:
    return {"data": item}


def __add_index_to_map_item(item) -> dict:
    key, value = item[0], item[1]
    return __convert_item_to_dict(value) | {"index": key}


def dataset_with_indices(
    dataset: torch.utils.data.Dataset,
) -> torch.utils.data.MapDataPipe:
    if isinstance(dataset, torch.utils.data.IterableDataset):
        return (
            torchdata.datapipes.iter.IterableWrapper(dataset)
            .map(__convert_item_to_dict)
            .add_index()
        )
    return torchdata.datapipes.map.Mapper(KeyPipe(dataset), __add_index_to_map_item)


def select_item(dataset, indices=None) -> Generator:
    if indices is not None:
        indices = set(indices)
    match dataset:
        case torch.utils.data.IterableDataset():
            if hasattr(dataset, "reset"):
                dataset.reset()
            iterator = iter(dataset)
            idx = 0
            for item in iterator:
                if indices is None or idx in indices:
                    yield idx, item
                    if indices is not None:
                        indices.remove(idx)
                idx += 1
            if hasattr(dataset, "reset"):
                dataset.reset()
        case _:
            if indices is None:
                indices = list(range(get_dataset_size(dataset)))
            for idx in indices:
                yield idx, dataset[idx]


def subset_dp(dataset, indices: None | list = None) -> torch.utils.data.MapDataPipe:
    return torchdata.datapipes.map.SequenceWrapper(
        list(dict(select_item(dataset, indices)).values()), deepcopy=False
    )


def get_iterable_item_key_and_value(item: Any) -> tuple:
    return item["index"], item["data"]


def convert_dataset_to_map_dp(
    dataset: torch.utils.data.IterableDataset,
) -> torch.utils.data.Dataset:
    dp = dataset_with_indices(dataset)
    if isinstance(dp, torch.utils.data.IterableDataset):
        return torchdata.datapipes.map.IterToMapConverter(
            dp, get_iterable_item_key_and_value
        )
    return dp
