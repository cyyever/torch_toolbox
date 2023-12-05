from typing import Any, Iterable, Mapping

import torch
import torch.utils.data
import torch.utils.data.datapipes
import torch.utils.data.dataset
from cyy_torch_toolbox.dependency import has_torch_geometric

if has_torch_geometric:
    import torch_geometric.data


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
    def __init__(self, dp: Mapping) -> None:
        super().__init__()
        self.__dp = dp

    def __getitem__(self, index) -> tuple:
        item = self.__dp.__getitem__(index)
        return (index, item)

    def __len__(self) -> int:
        return len(self.__dp)


def __add_index_to_map_item(item) -> dict:
    key, value = item[0], item[1]
    return {"index": key, "data": value}


def dataset_with_indices(
    dataset: torch.utils.data.Dataset,
) -> torch.utils.data.MapDataPipe:
    old_dataset = dataset
    if has_torch_geometric:
        match dataset:
            case torch_geometric.data.Dataset():
                return dataset
    match dataset:
        case list():
            return dataset
        case torch.utils.data.IterableDataset():
            dataset = torch.utils.data.datapipes.iter.IterableWrapper(dataset)
    # if has_hugging_face:
    #     if isinstance(dataset, hugging_face_datasets.arrow_dataset.Dataset):
    #         return dataset
    # dataset = torchdata.datapipes.iter.IterableWrapper(dataset)
    match dataset:
        case torch.utils.data.IterDataPipe():
            dataset = dataset.enumerate()
        case _:
            dataset = torch.utils.data.datapipes.map.Mapper(
                KeyPipe(dataset), __add_index_to_map_item
            )
    assert not hasattr(dataset, "original_dataset")
    setattr(dataset, "original_dataset", old_dataset)
    return dataset


def select_item(
    dataset: Any,
    indices: None | Iterable = None,
    mask: None | list[torch.Tensor] = None,
) -> Iterable:
    if indices is not None:
        indices = set(indices)
    if has_torch_geometric:
        match dataset:
            case torch_geometric.data.Dataset() | [
                torch_geometric.data.Dataset(),
                *_,
            ] | [{"original_dataset": torch_geometric.data.Dataset()}, *_]:
                if mask is None:
                    for idx, data in enumerate(dataset):
                        yield idx, data
                    return
                assert len(mask) == 1
                if isinstance(dataset, torch_geometric.data.Dataset):
                    for idx, flag in enumerate(mask[0].tolist()):
                        if not flag:
                            continue
                        if indices is None or idx in indices:
                            yield idx, {"target": dataset[0].y[idx], "index": idx}
                else:
                    graph = dataset[0]["original_dataset"][dataset[0]["graph_index"]]
                    for idx, flag in enumerate(mask[0].tolist()):
                        if not flag:
                            continue
                        if indices is None or idx in indices:
                            yield idx, {
                                "target": graph.y[idx],
                                "index": idx,
                            }
                return

    match dataset:
        case torch.utils.data.IterableDataset():
            if hasattr(dataset, "reset"):
                dataset.reset()
            iterator = iter(dataset)
            for idx, item in enumerate(iterator):
                if indices is None or idx in indices:
                    yield idx, item
                    if indices is not None:
                        indices.remove(idx)
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
    # original_dataset = getattr(dataset, "dataset", None)
    # if has_hugging_face:
    #     match original_dataset:
    #         case hugging_face_datasets.arrow_dataset.Dataset():
    #             pass

    return torch.utils.data.datapipes.map.SequenceWrapper(
        list(dict(select_item(dataset, indices)).values()), deepcopy=False
    )
