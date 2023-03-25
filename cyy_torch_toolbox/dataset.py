from typing import Generator

from cyy_torch_toolbox.dependency import has_torch_geometric

if has_torch_geometric:
    import torch_geometric
    import torch_geometric.data.dataset

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
    old_dataset = dataset
    if has_torch_geometric:
        if isinstance(
            dataset,
            torch_geometric.data.dataset.Dataset | torch_geometric.data.dataset.Data,
        ):
            return dataset
    if isinstance(dataset, torch.utils.data.IterableDataset):
        dataset = torchdata.datapipes.iter.IterableWrapper(dataset)
    if isinstance(dataset, torchdata.datapipes.iter.IterDataPipe):
        dataset = dataset.enumerate()
    else:
        dataset = torchdata.datapipes.map.Mapper(
            KeyPipe(dataset), __add_index_to_map_item
        )
    assert not hasattr(dataset, "dataset")
    setattr(dataset, "dataset", old_dataset)
    return dataset


def select_item(dataset, indices=None) -> Generator:
    if indices is not None:
        indices = set(indices)
    if has_torch_geometric:
        match dataset:
            case torch_geometric.data.dataset.Data():
                for idx, mask in enumerate(dataset.train_mask.tolist()):
                    if not mask:
                        continue
                    if indices is None or idx in indices:
                        yield idx, {"target": dataset.y[idx], "index": idx}
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


def subset_dp(dataset, indices: None | list = None) -> torch.utils.data.MapDataPipe:
    return torchdata.datapipes.map.SequenceWrapper(
        list(dict(select_item(dataset, indices)).values()), deepcopy=False
    )
