from collections.abc import Callable, Iterable, Mapping
from typing import Any, Self

import torch
import torch.utils.data
from cyy_naive_lib.log import log_error
from torch.utils.data import default_collate

from .common import DataExtraction
from .dataset import select_item
from .transform import BatchTransform, Transform


class DataPipeline:
    def __init__(self, transforms: list[Transform] | None = None) -> None:
        self.__transforms: list[Transform] = []
        if transforms is not None:
            self.__transforms = transforms
        else:
            self.append(DataExtraction())

    @property
    def transforms(self) -> list[Transform]:
        return self.__transforms

    def __len__(self) -> int:
        return len(self.__transforms)

    def has_transform(self, name: str) -> bool:
        return any(t.name == name for t in self.__transforms)

    def clear(self) -> None:
        self.__transforms = []

    def append(self, transform: Transform) -> None:
        self.__transforms.append(transform)

    def prepend(self, transform: Transform) -> None:
        self.__transforms.insert(0, transform)

    def is_valid(self) -> bool:
        has_for_batch = False
        for t in self.__transforms:
            if has_for_batch and not isinstance(t, BatchTransform):
                return False
            if isinstance(t, BatchTransform):
                has_for_batch = True
        return True

    def cache(self, data: Any) -> tuple[Any, Self]:
        return self.__apply_until(
            data,
            lambda t: t.cacheable and not isinstance(t, BatchTransform),
        )

    def __apply_until(
        self, data: Any, cond: Callable | None = None
    ) -> tuple[Any, Self]:
        assert self.is_valid()
        for idx, t in enumerate(self.__transforms):
            if cond is not None and not cond(t):
                return data, type(self)(transforms=self.__transforms[idx:])
            data = t(data)
        return data, type(self)(transforms=[])

    def apply_first(self, data: Any) -> Any:
        assert self.__transforms
        return self.__transforms[0](data)

    def apply(self, data: Any) -> tuple[Any, Self]:
        return self.__apply_until(data, lambda t: not isinstance(t, BatchTransform))

    def apply_batch(self, data: Any) -> Any:
        if not self.__transforms:
            return data
        assert isinstance(self.__transforms[0], BatchTransform)
        res, remaining_pipeline = self.__apply_until(data)
        assert len(remaining_pipeline) == 0
        return res

    def cache_dataset(self, dataset: torch.utils.data.Dataset) -> tuple[list, Self]:
        transformed_dataset: list = []
        remaining_pipeline: None | Self = None
        for _, item in select_item(dataset):
            item, remaining_pipeline = self.apply(item)
            transformed_dataset.append(item)
        assert remaining_pipeline is not None
        return transformed_dataset, remaining_pipeline

    def collate_batch(self, batch: Iterable) -> dict:
        batch_size = 0
        result: dict | None | list = None
        batch_transforms: None | Self = None
        for data in batch:
            data, batch_transforms = self.apply(data)
            match data:
                case dict():
                    if result is None:
                        result = {}
                    assert isinstance(result, dict)
                    for k, v in data.items():
                        if k not in result:
                            result[k] = []
                        result[k].append(v)
                case _:
                    if result is None:
                        result = []
                    assert isinstance(result, list)
                    result.append(data)
            batch_size += 1
        assert result is not None
        assert batch_transforms is not None
        if isinstance(result, list):
            result = {"input": result}
        result = batch_transforms.apply_batch(result)
        assert result is not None
        assert isinstance(result, Mapping)
        for k, v in result.items():
            if isinstance(v, list) and k in ("index", "input", "target"):
                try:
                    collated_value = default_collate(v)
                    result[k] = collated_value
                except BaseException:
                    pass

        result["batch_size"] = batch_size
        if "index" in result:
            result["sample_indices"] = result.pop("index")
        if "input" in result:
            result["inputs"] = result.pop("input")
        if "target" in result:
            result["targets"] = result.pop("target")
        return result

    def __str__(self) -> str:
        return "\n".join(str(f) for f in self.__transforms)
