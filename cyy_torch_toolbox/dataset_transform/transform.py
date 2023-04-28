import copy
from typing import Any, Callable

from cyy_naive_lib.log import get_logger
from torch.utils.data._utils.collate import default_collate

from ..dataset import select_item
from ..ml_type import TransformType
from ..tensor import tensor_to
from .common import default_data_extraction


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

    def get(self, key: TransformType) -> list:
        return self.__transforms.get(key, [])

    def get_input_transforms_in_order(self, include_random: bool = True) -> list:
        res = self.get(TransformType.InputText) + self.get(TransformType.Input)
        if include_random:
            res += self.get(TransformType.RandomInput)
        return res

    def get_target_transforms_in_order(self) -> list:
        return self.get(TransformType.Target)

    def transform_text(self, text):
        for f in self.get(TransformType.InputText):
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

    def transform_inputs(self, inputs: list) -> list:
        batch_transforms = self.get(TransformType.InputBatch)
        if not batch_transforms:
            batch_transforms.append(default_collate)
        for f in batch_transforms:
            inputs = f(inputs)
        return inputs

    def transform_target(self, target: Any, index=None) -> Any:
        for f in self.get(TransformType.Target):
            target = f(target, index)
        return target

    def transform_targets(self, targets: Any) -> Any:
        batch_transforms = self.get(TransformType.TargetBatch)
        if not batch_transforms:
            batch_transforms.append(default_collate)
        for f in batch_transforms:
            targets = f(targets)
        return targets.reshape(-1)

    def collate_batch(self, batch: Any) -> dict:
        inputs = []
        targets = []
        other_info: list | dict = []
        for data in batch:
            data = copy.copy(self.extract_data(data))
            sample_input = self.transform_input(data.pop("input"))
            inputs.append(sample_input)
            targets.append(
                self.transform_target(
                    target=data.pop("target"), index=data.get("index", None)
                )
            )
            other_info.append(data)
        inputs = self.transform_inputs(inputs)
        targets = self.transform_targets(targets)
        if other_info:
            other_info = default_collate(other_info)
            assert isinstance(other_info, dict)
            if "index" in other_info:
                other_info["sample_indices"] = other_info.pop("index")
        else:
            other_info = {}
        batch_size = len(batch)

        if (
            hasattr(inputs, "device")
            and hasattr(targets, "device")
            and targets.device != inputs.device
        ):
            targets = tensor_to(targets, device=inputs.device, non_blocking=False)
        assert "batch_size" not in other_info
        return {
            "batch_size": batch_size,
            "inputs": inputs,
            "targets": targets,
        } | other_info

    def cache_transforms(
        self, dataset: Any, device: Any | None = None
    ) -> tuple[dict, Any]:
        if device is not None:
            get_logger().warning("cache dataset to device memory: %s", device)
        else:
            get_logger().warning("cache dataset to main memory")
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
        new_transforms.clear(TransformType.Input)
        new_transforms.clear(TransformType.Target)
        return transformed_dataset, new_transforms

    def __str__(self):
        desc = []
        for k in (
            TransformType.ExtractData,
            TransformType.InputText,
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
