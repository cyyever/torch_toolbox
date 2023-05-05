from collections.abc import Iterable
from typing import Generator

from cyy_torch_toolbox.dependency import has_torch_geometric

if has_torch_geometric:
    import torch_geometric.data

import torch
import torchdata


def get_dataset_size(dataset: torch.utils.data.Dataset) -> int:
    if has_torch_geometric:
        match dataset:
            case [torch_geometric.data.Data()]:
                graph = dataset[0]
                return graph["mask"].sum()
            case [{"subset_mask": subset_mask, "graph": _}]:
                return subset_mask.sum()
    match dataset:
        case torch.utils.data.dataset.ConcatDataset():
            return sum(get_dataset_size(d) for d in dataset.datasets)
        case torch.utils.data.IterableDataset():
            cnt: int = 0
            for _ in dataset:
                cnt += 1
            return cnt
        case torchdata.datapipes.map.MapDataPipe():
            return len(dataset)
    raise NotImplementedError(dataset)


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
    old_dataset = dataset
    if has_torch_geometric:
        if isinstance(dataset, torch_geometric.data.dataset.Dataset):
            return dataset
    match dataset:
        case torch.utils.data.IterableDataset():
            dataset = torchdata.datapipes.iter.IterableWrapper(dataset)
    match dataset:
        case torchdata.datapipes.iter.IterDataPipe():
            dataset = dataset.enumerate()
        case _:
            dataset = torchdata.datapipes.map.Mapper(
                KeyPipe(dataset), __add_index_to_map_item
            )
    assert not hasattr(dataset, "dataset")
    setattr(dataset, "dataset", old_dataset)
    return dataset


def select_item(
    dataset: torch.utils.data.Dataset,
    indices: None | Iterable = None,
    mask: None | torch.Tensor = None,
) -> Generator:
    if indices is not None:
        indices = set(indices)
    if has_torch_geometric:
        match dataset:
            case torch_geometric.data.Dataset() | list():
                if mask is None:
                    idx = 0
                    for data in dataset:
                        yield idx, data
                        idx += 1
                    return
                if isinstance(dataset, torch_geometric.data.Dataset):
                    for idx, flag in enumerate(mask.tolist()):
                        if not flag:
                            continue
                        if indices is None or idx in indices:
                            yield idx, {"target": dataset.data.y[idx], "index": idx}
                else:
                    for idx, flag in enumerate(mask.tolist()):
                        if not flag:
                            continue
                        if indices is None or idx in indices:
                            yield idx, {
                                "target": dataset[0]["graph"]["y"][idx],
                                "index": idx,
                            }
                return

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


def subset_dp(
    dataset: torch.utils.data.Dataset, indices: None | Iterable = None
) -> torch.utils.data.MapDataPipe:
    return torchdata.datapipes.map.SequenceWrapper(
        list(dict(select_item(dataset, indices)).values()), deepcopy=False
    )
