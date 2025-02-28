from collections.abc import Generator
from typing import Any

import torch
import torch.utils.data
import torch.utils.data.datapipes
import torch.utils.data.dataset

from ..ml_type import OptionalIndicesType
from .transform import DatasetTransform


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


def select_item(dataset: Any, indices: OptionalIndicesType = None) -> Generator:
    if indices is not None:
        indices = set(
            int(idx.item()) if isinstance(idx, torch.Tensor) else idx for idx in indices
        )
    match dataset:
        case torch.utils.data.IterableDataset():
            iterator = iter(dataset)
            if indices is None:
                yield from enumerate(iterator)
            else:
                for idx, item in enumerate(iterator):
                    if idx in indices:
                        indices.remove(idx)
                        yield idx, item
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


class KeyPipe(torch.utils.data.MapDataPipe):
    def __init__(self, dp: Any) -> None:
        super().__init__()
        self.__dp = dp

    def __getitem__(self, index) -> tuple:
        item = self.__dp[index]
        return (index, item)

    def __len__(self) -> int:
        return len(self.__dp)


class DatasetWithIndex(DatasetTransform):
    def __init__(self) -> None:
        super().__init__(fun=DatasetWithIndex.apply, name="add index to dataset")

    @classmethod
    def apply(cls, data: Any) -> Any:
        dataset = data
        return torch.utils.data.datapipes.map.Mapper(
            KeyPipe(dataset), DatasetWithIndex._add_index_to_map_item
        )

    @classmethod
    def _add_index_to_map_item(cls, item) -> dict:
        key, value = item[0], item[1]
        return {"index": key, "data": value}
