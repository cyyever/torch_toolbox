import copy
import functools
from collections.abc import Iterable
from typing import Any, Generator, Type

import torch
import torch.nn.functional
import torch.utils.data

from ..data_pipeline.dataset import get_dataset_size, select_item, subset_dp
from ..data_pipeline.transform import Transforms
from ..factory import Factory


class DatasetUtil:
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        name: None | str = None,
        transforms: Transforms | None = None,
        cache_dir: None | str = None,
    ) -> None:
        self.dataset: torch.utils.data.Dataset = dataset
        self.__len: None | int = None
        self._name: str = name if name else ""
        self.__transforms: Transforms | None = transforms
        self._cache_dir = cache_dir

    def __len__(self) -> int:
        if self.__len is None:
            self.__len = get_dataset_size(self.dataset)
        return self.__len

    def decompose(self) -> None | dict:
        return None

    def get_subset(self, indices: Iterable) -> torch.utils.data.MapDataPipe:
        return subset_dp(self.dataset, indices)

    def get_raw_samples(self, indices: Iterable | None = None) -> Iterable:
        return select_item(dataset=self.dataset, indices=indices)

    def get_samples(self, indices: Iterable | None = None) -> Iterable:
        raw_samples = self.get_raw_samples(indices=indices)
        if self.__transforms is None:
            return raw_samples
        for idx, sample in raw_samples:
            sample = self.__transforms.extract_data(sample)
            yield idx, sample

    def get_sample(self, index: int) -> Any:
        for _, sample in self.get_samples(indices=[index]):
            return sample
        return None

    @classmethod
    def __decode_target(cls: Type, target: Any) -> set:
        match target:
            case int() | str():
                return {target}
            case torch.Tensor():
                if target.numel() == 1:
                    return {target.item()}
                # one hot vector
                if (0 <= target <= 1).all().item():
                    return set(target.nonzero().view(-1).tolist())
                raise NotImplementedError(f"Unsupported target {target}")
            case dict():
                if "labels" in target:
                    return cls.__decode_target(target["labels"].tolist())
                if all(isinstance(s, str) and s.isnumeric() for s in target):
                    return cls.__decode_target({int(s) for s in target})
            case Iterable():
                return set(target)
        raise RuntimeError("can't extract labels from target: " + str(target))

    @classmethod
    def replace_target(cls, old_target: Any, new_target: dict) -> Any:
        match old_target:
            case int() | str():
                old_target_value = list(cls.__decode_target(old_target))[0]
                new_target_value = new_target.get(old_target_value, None)
                if new_target_value is None:
                    return old_target
                assert len(new_target_value) == 1
                return type(old_target)(list(new_target_value)[0])
            case torch.Tensor():
                old_target_value = cls.__decode_target(old_target)
                if old_target.numel() == 1:
                    old_target_value = list(old_target_value)[0]
                    if old_target_value not in new_target:
                        return old_target
                    new_target_tensor = old_target.clone()
                    new_target_tensor = new_target[old_target_value]
                    return new_target_tensor
                # one hot vector
                if (0 <= old_target <= 1).all().item():
                    old_target_value = {
                        new_target.get(old_t, old_t) for old_t in old_target_value
                    }
                    return torch.nn.functional.one_hot(
                        torch.tensor(list(old_target_value)),
                        num_classes=old_target.shape[-1],
                    )
                raise NotImplementedError(f"Unsupported target {old_target}")
            case dict():
                if "labels" in old_target:
                    new_target_dict = copy.deepcopy(old_target)
                    new_target_dict["labels"] = cls.replace_target(
                        new_target_dict["labels"], new_target
                    )
                    return new_target_dict
            case Iterable():
                old_target_value = cls.__decode_target(old_target)
                return type(old_target)(
                    new_target.get(old_t, old_t) for old_t in old_target_value
                )

        raise RuntimeError(f"can't convert labels {new_target} for target {old_target}")

    def _get_sample_input(self, index: int, apply_transform: bool = True) -> Any:
        sample = self.get_sample(index)
        sample_input = sample["input"]
        if apply_transform:
            assert self.__transforms is not None
            sample_input = self.__transforms.transform_input(
                sample_input, apply_random=False
            )
        return sample_input

    def get_batch_labels(self, indices: None | Iterable[int] = None) -> Generator:
        for idx, sample in self.get_samples(indices):
            target = sample["target"]
            if self.__transforms is not None:
                target = self.__transforms.transform_target(target)
            yield idx, DatasetUtil.__decode_target(target)

    def get_sample_label(self, index: int) -> Any:
        labels = list(self.get_batch_labels(indices=[index]))[0][1]
        assert len(labels) == 1
        return next(iter(labels))

    def get_labels(self) -> set:
        return set().union(*tuple(set(labels) for _, labels in self.get_batch_labels()))

    def get_original_dataset(self) -> torch.utils.data.Dataset:
        return self.dataset[0].get("original_dataset", self.dataset)

    def get_label_names(self) -> dict:
        original_dataset = self.get_original_dataset()
        if (
            hasattr(original_dataset, "classes")
            and original_dataset.classes
            and isinstance(original_dataset.classes[0], str)
        ):
            return dict(enumerate(original_dataset.classes))

        def get_label_name(container: set, index: int) -> set:
            label = self.get_sample_label(index)
            if isinstance(label, str):
                container.add(label)
            return container

        label_names: set = functools.reduce(get_label_name, range(len(self)), set())
        if label_names:
            return dict(enumerate(sorted(label_names)))
        raise RuntimeError("no label names detected")


global_dataset_util_factor = Factory()
