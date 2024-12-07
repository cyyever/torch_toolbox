import copy
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Self

import torch
import torch.utils.data
from cyy_naive_lib.log import log_debug
from torch.utils.data import default_collate

from ..ml_type import TransformType
from ..tensor import tensor_to
from .common import default_data_extraction
from .dataset import select_item


class Transforms:
    def __init__(self) -> None:
        self.__transforms: dict = {}
        self.append(key=TransformType.ExtractData, transform=default_data_extraction)

    def has_transform(self) -> bool:
        return any(self.__transforms.values())

    def clear(self, key: TransformType) -> None:
        self.__transforms.pop(key, None)

    def append(self, key: TransformType, transform: Callable) -> None:
        if key not in self.__transforms:
            self.__transforms[key] = []
        self.__transforms[key].append(transform)

    def set_one(self, key: TransformType, transform: Callable) -> None:
        assert key not in self.__transforms
        self.append(key=key, transform=transform)

    def get(self, key: TransformType) -> list:
        return self.__transforms.get(key, [])

    def get_input_transforms_in_order(self, include_random: bool = True) -> list:
        res = (
            self.get(TransformType.InputText)
            + self.get(TransformType.InputTextLast)
            + self.get(TransformType.Input)
        )
        if include_random:
            res += self.get(TransformType.RandomInput)
        return res

    def get_target_transforms_in_order(self) -> list:
        return self.get(TransformType.Target)

    def transform_text(self, text):
        for f in self.get(TransformType.InputText) + self.get(
            TransformType.InputTextLast
        ):
            text = f(text)
        return text

    def extract_data(self, data):
        for f in self.get(TransformType.ExtractData):
            data = f(data)
        return data

    def transform_input(self, sample_input: Any, apply_random: bool = True) -> Any:
        for f in self.get_input_transforms_in_order(include_random=apply_random):
            sample_input = f(sample_input)
        return sample_input

    def transform_inputs(self, inputs: Iterable) -> Any:
        batch_transforms = self.get(TransformType.InputBatch)
        if not batch_transforms:
            batch_transforms.append(default_collate)
        for f in batch_transforms:
            inputs = f(inputs)
        return inputs

    def transform_target(self, target: Any, index: int | None = None) -> Any:
        for f in self.get(TransformType.Target):
            target = f(target, index)
        return target

    def transform_targets(self, targets: Iterable) -> Any:
        batch_transforms = self.get(TransformType.TargetBatch)
        if not batch_transforms:
            batch_transforms.append(default_collate)
        for f in batch_transforms:
            targets = f(targets)
        return targets

    def collate_batch(self, batch: Iterable) -> dict:
        inputs: list = []
        targets = []
        other_info: list = []
        for data in batch:
            data = copy.copy(self.extract_data(data))
            if "target" in data:
                targets.append(
                    self.transform_target(
                        target=data.pop("target"), index=data.get("index", None)
                    )
                )
            if "input" in data:
                sample_input = self.transform_input(data.pop("input"))
                other_info.append(data)
            else:
                data = self.transform_input(data)
                sample_input = data
            inputs.append(sample_input)
        batch_size = len(inputs)
        transformed_inputs = self.transform_inputs(inputs)
        res = {
            "batch_size": batch_size,
            "inputs": transformed_inputs,
        }
        if targets:
            targets = self.transform_targets(targets)
            res["targets"] = targets
        elif "labels" in transformed_inputs:
            targets = transformed_inputs.pop("labels")
            res["targets"] = targets
        if other_info:
            tmp: dict = default_collate(other_info)
            assert isinstance(tmp, dict)
            if "index" in tmp:
                tmp["sample_indices"] = tmp.pop("index")
            res |= tmp
        return res

    def cache_transforms(
        self, dataset: torch.utils.data.Dataset, device: torch.device
    ) -> tuple[dict, Self]:
        log_debug("cache dataset to device: %s", device)
        transformed_dataset: dict = {}
        for k, item in select_item(dataset):
            item = self.extract_data(item)
            item["input"] = self.transform_input(item["input"], apply_random=False)
            item["target"] = self.transform_target(item["target"], index=k)
            if device is not None:
                item["input"] = tensor_to(
                    item["input"], device=device, non_blocking=True
                )
                item["target"] = tensor_to(
                    item["target"], device=device, non_blocking=True
                )
            transformed_dataset[k] = item
        new_transforms = copy.deepcopy(self)
        new_transforms.clear(TransformType.ExtractData)
        new_transforms.append(
            key=TransformType.ExtractData, transform=default_data_extraction
        )
        new_transforms.clear(TransformType.InputText)
        new_transforms.clear(TransformType.InputTextLast)
        new_transforms.clear(TransformType.Input)
        new_transforms.clear(TransformType.Target)
        return transformed_dataset, new_transforms

    def __str__(self) -> str:
        desc = []
        for k in (
            TransformType.ExtractData,
            TransformType.InputText,
            TransformType.InputTextLast,
            TransformType.Input,
            TransformType.RandomInput,
            TransformType.InputBatch,
            TransformType.Target,
        ):
            transforms = self.__transforms.get(k, [])
            if transforms:
                desc.append(str(k) + "=>")
                for t in transforms:
                    desc.append(str(t))
        return "\n".join(desc)


@dataclass(kw_only=True)
class Transform:
    fun: Callable
    name: str = ""
    cacheable: bool = False
    component: str | None = None
    for_batch: bool = False

    def __str__(self) -> str:
        fun_name = self.name if self.name else str(self.fun)
        return f"name:{fun_name} cacheable:{self.cacheable}"

    def __call__(self, data: Any) -> Any:
        if self.component is not None:
            data[self.component] = self.fun(data[self.component])
            return data
        return self.fun(data)


class DataPipeline:
    def __init__(self, transforms: list[Transform] | None = None) -> None:
        self.__transforms: list[Transform] = []
        if transforms is not None:
            self.__transforms = transforms
        else:
            self.append(
                transform=Transform(
                    fun=default_data_extraction, name="data_extraction", cacheable=True
                )
            )

    def __len__(self) -> int:
        return len(self.__transforms)

    def clear(self) -> None:
        self.__transforms = []

    def append(self, transform: Transform) -> None:
        self.__transforms.append(transform)

    def is_valid(self) -> bool:
        has_for_batch = False
        for t in self.__transforms:
            if has_for_batch and not t.for_batch:
                return False
            if t.for_batch:
                has_for_batch = True
        return True

    def cache(self, data: Any) -> tuple[Any, Self]:
        return self.__apply_until(data, lambda t: t.cacheable and not t.for_batch)

    def __apply_until(
        self, data: Any, cond: Callable | None = None
    ) -> tuple[Any, Self]:
        assert self.is_valid()
        for idx, t in enumerate(self.__transforms):
            if cond is not None and not cond(t):
                return data, type(self)(transforms=self.__transforms[idx:])
            data = t(data)
        return data, type(self)(transforms=[])

    def apply(self, data: Any) -> tuple[Any, Self]:
        return self.__apply_until(data, lambda t: not t.for_batch)

    def apply_batch(self, data: Any) -> Any:
        if not self.__transforms:
            return data
        assert self.__transforms[0].for_batch
        res, remaining_pipeline = self.__apply_until(data)
        assert len(remaining_pipeline) == 0
        return res

    def cache_dataset(self, dataset: torch.utils.data.Dataset) -> tuple[list, Self]:
        transformed_dataset: list = []
        remaining_pipeline: None | Self = None
        for _, item in select_item(dataset):
            item, remaining_pipeline = self.apply(item)
            transformed_dataset.append(item)
        assert transformed_dataset is not None
        return transformed_dataset, remaining_pipeline

    def collate_batch(self, batch: Iterable) -> dict:
        batch_size = 0
        result: dict | None = None
        batch_transforms: None | Self = None
        for data in batch:
            data, batch_transforms = self.apply(data)
            if result is None:
                result = {}
            for k, v in data.items():
                if k not in result:
                    result[k] = []
                result[k].append(v)
            batch_size += 1
        assert batch_transforms is not None
        result = batch_transforms.apply_batch(result)
        assert result is not None
        for k, v in result.items():
            if not isinstance(v, torch.Tensor):
                result[k] = default_collate(v)

        assert isinstance(result, dict)
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
