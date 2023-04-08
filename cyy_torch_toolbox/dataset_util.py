import copy
import functools
import os
import random
from typing import Any, Generator

import PIL
import torch
import torch.utils
from cyy_naive_lib.log import get_logger

from cyy_torch_toolbox.dataset import get_dataset_size, select_item, subset_dp
from cyy_torch_toolbox.dataset_transform.transforms import Transforms
from cyy_torch_toolbox.dependency import has_torch_geometric, has_torchvision
from cyy_torch_toolbox.ml_type import MachineLearningPhase

if has_torchvision:
    import torchvision
if has_torch_geometric:
    import torch_geometric.data


class DatasetUtil:
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        name: None | str = None,
        transforms: Transforms | None = None,
        phase: MachineLearningPhase | None = None,
    ) -> None:
        self.dataset: torch.utils.data.Dataset = dataset
        self.__len: None | int = None
        self._name: str | None = name
        self.__transforms: Transforms | None = transforms
        self._phase = phase

    def __len__(self) -> int:
        if self.__len is None:
            self.__len = get_dataset_size(self.dataset)
        return self.__len

    def get_samples(self, indices=None) -> Generator:
        items = select_item(dataset=self.dataset, indices=indices, mask=self.get_mask())
        if self.__transforms is None:
            return items
        for idx, sample in items:
            sample = self.__transforms.extract_data(sample)
            yield idx, sample

    def get_mask(self):
        return None

    def get_sample(self, index: int) -> Any:
        for _, sample in self.get_samples(indices=[index]):
            return sample
        return None

    @classmethod
    def __decode_target(cls, target) -> set:
        match target:
            case int() | str():
                return set([target])
            case list():
                return set(target)
            case torch.Tensor():
                return cls.__decode_target(target.tolist())
            case dict():
                if "labels" in target:
                    return set(target["labels"].tolist())
                if all(isinstance(s, str) and s.isnumeric() for s in target):
                    return set(int(s) for s in target)
        raise RuntimeError("can't extract labels from target: " + str(target))

    def _get_sample_input(self, index: int, apply_transform: bool = True) -> Any:
        sample = self.get_sample(index)
        sample_input = sample["input"]
        if apply_transform:
            assert self.__transforms is not None
            sample_input = self.__transforms.transform_input(
                sample_input, apply_random=False
            )
        return sample_input

    def get_batch_labels(self, indices=None) -> Generator:
        for idx, sample in self.get_samples(indices):
            target = sample["target"]
            if self.__transforms is not None:
                target = self.__transforms.transform_target(target)
            yield idx, DatasetUtil.__decode_target(target)

    def get_sample_label(self, index):
        for _, labels in self.get_batch_labels(indices=[index]):
            assert len(labels) == 1
            return next(iter(labels))
        return None

    def get_labels(self) -> set:
        return set().union(*tuple(set(labels) for _, labels in self.get_batch_labels()))

    def get_original_dataset(self) -> torch.utils.data.Dataset:
        dataset = self.dataset
        if hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        return dataset

    def get_label_names(self) -> dict:
        original_dataset = self.get_original_dataset()
        if hasattr(original_dataset, "classes"):
            classes = getattr(original_dataset, "classes")
            if classes and isinstance(classes[0], str):
                return dict(enumerate(classes))

        def get_label_name(container: set, index) -> set:
            label = self.get_sample_label(index)
            if isinstance(label, str):
                container.add(label)
            return container

        label_names = functools.reduce(get_label_name, range(len(self)), set())
        if label_names:
            return dict(enumerate(sorted(label_names)))
        raise RuntimeError("no label names detected")


class DatasetSplitter(DatasetUtil):
    __sample_label_dict: None | dict = None
    __label_sample_dict: None | dict = None

    @property
    def sample_label_dict(self) -> dict[int, list]:
        if self.__sample_label_dict is not None:
            return self.__sample_label_dict
        self.__sample_label_dict = dict(self.get_batch_labels())
        return self.__sample_label_dict

    @property
    def label_sample_dict(self) -> dict:
        if self.__label_sample_dict is not None:
            return self.__label_sample_dict
        self.__label_sample_dict = {}
        for index, labels in self.sample_label_dict.items():
            for label in labels:
                if label not in self.__label_sample_dict:
                    self.__label_sample_dict[label] = [index]
                else:
                    self.__label_sample_dict[label].append(index)
        return self.__label_sample_dict

    def iid_split_indices(self, parts: list) -> list:
        return self.__get_split_indices(parts, iid=True)

    def random_split_indices(self, parts: list) -> list:
        return self.__get_split_indices(parts, iid=False)

    def iid_split(self, parts: list) -> list:
        return self.split_by_indices(self.iid_split_indices(parts))

    def get_subset(self, indices):
        return subset_dp(self.dataset, indices)

    def split_by_indices(self, indices_list: list) -> list:
        return [self.get_subset(indices) for indices in indices_list]

    def __get_split_indices(self, parts: list, iid: bool = True) -> list[list]:
        assert parts
        if len(parts) == 1:
            return [list(range(len(self)))]

        def split_index_impl(indices_list: list) -> list[list]:
            part_lens = []
            for part in parts:
                part_len = int(len(indices_list) * part / sum(parts))
                assert part_len != 0
                part_lens.append(part_len)
            part_lens[-1] += len(indices_list) - sum(part_lens)
            part_indices = []
            for part_len in part_lens:
                part_indices.append(indices_list[0:part_len])
                indices_list = indices_list[part_len:]
            return part_indices

        if not iid:
            index_list = list(range(len(self)))
            random.shuffle(index_list)
            return split_index_impl(index_list)

        sub_index_list: list[list] = []
        for _ in parts:
            sub_index_list.append([])
        for v in self.label_sample_dict.values():
            part_index_list = split_index_impl(sorted(v))
            random.shuffle(part_index_list)
            for a, b in zip(sub_index_list, part_index_list):
                a += b
        return sub_index_list

    def sample_by_labels(self, percents: list[float]) -> dict:
        sample_indices: dict = {}
        for idx, label in enumerate(sorted(self.label_sample_dict.keys())):
            v = self.label_sample_dict[label]
            sample_size = int(len(v) * percents[idx])
            if sample_size == 0:
                sample_indices[label] = []
            else:
                sample_indices[label] = random.sample(v, k=sample_size)
        return sample_indices

    def iid_sample(self, percentage: float) -> dict:
        return self.sample_by_labels([percentage] * len(self.label_sample_dict))

    def randomize_subset_label(self, percentage: float) -> dict:
        sample_indices = self.iid_sample(percentage)
        labels = self.get_labels()
        randomized_label_map = {}
        for label, indices in sample_indices.items():
            other_labels = list(set(labels) - set([label]))
            for index in indices:
                randomized_label_map[index] = random.choice(other_labels)
        return randomized_label_map


class VisionDatasetUtil(DatasetSplitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__channel = None

    @property
    def channel(self):
        if self.__channel is not None:
            return self.__channel
        x = self._get_sample_input(0)
        self.__channel = x.shape[0]
        assert self.__channel <= 3
        return self.__channel

    def get_mean_and_std(self):
        if self._name is not None and self._name.lower() == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            return (mean, std)
        mean = torch.zeros(self.channel)
        for index in range(len(self)):
            x = self._get_sample_input(index)
            for i in range(self.channel):
                mean[i] += x[i, :, :].mean()
        mean.div_(len(self))

        wh = None
        std = torch.zeros(self.channel)
        for index in range(len(self)):
            x = self._get_sample_input(index)
            if wh is None:
                wh = x.shape[1] * x.shape[2]
            for i in range(self.channel):
                std[i] += torch.sum((x[i, :, :] - mean[i].data.item()) ** 2) / wh
        std = std.div(len(self)).sqrt()
        return mean, std

    def save_sample_image(self, index: int, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sample_input = self._get_sample_input(index, apply_transform=False)
        match sample_input:
            case PIL.Image.Image():
                sample_input.save(path)
            case _:
                torchvision.utils.save_image(sample_input, path)

    @torch.no_grad()
    def get_sample_image(self, index: int) -> Any:
        tensor = self._get_sample_input(index, apply_transform=False)
        if isinstance(tensor, PIL.Image.Image):
            return tensor
        grid = torchvision.utils.make_grid(tensor)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = (
            grid.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )
        return PIL.Image.fromarray(ndarr)


class TextDatasetUtil(DatasetSplitter):
    @torch.no_grad()
    def get_sample_text(self, index: int) -> str:
        return self._get_sample_input(index, apply_transform=False)


class GraphDatasetUtil(DatasetSplitter):
    def get_mask(self):
        mask = None
        assert len(self.dataset) == 1
        match self._phase:
            case MachineLearningPhase.Training:
                mask = self.dataset[0].train_mask
            case MachineLearningPhase.Validation:
                mask = self.dataset[0].val_mask
            case MachineLearningPhase.Test:
                mask = self.dataset[0].test_mask
            case _:
                raise NotImplementedError()
        return mask

    def get_subset(self, indices):
        mask = copy.deepcopy(self.get_mask())
        dataset = copy.deepcopy(self.dataset)
        mask.fill_(False)
        for index in indices:
            mask[index] = True
        data_dict = dataset[0].to_dict()
        match self._phase:
            case MachineLearningPhase.Training:
                assert "train_mask" in data_dict
                data_dict["train_mask"] = mask
            case MachineLearningPhase.Validation:
                assert "val_mask" in data_dict
                data_dict["val_mask"] = mask
            case MachineLearningPhase.Test:
                assert "test_mask" in data_dict
                data_dict["test_mask"] = mask
            case _:
                raise NotImplementedError()
        dataset = [torch_geometric.data.Data.from_dict(data_dict)]
        return dataset
