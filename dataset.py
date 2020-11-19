import os
import functools
from typing import Iterable, Callable, List, Generator
import random
import PIL


import torch
import torchvision
import torchvision.transforms as transforms

from cyy_naive_lib.log import get_logger
from datasets.webank_street_dataset import WebankStreetDataset
from local_types import MachineLearningPhase


class DatasetFilter:
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            filters: Iterable[Callable]):
        self.dataset = dataset
        self.filters = filters
        self.indices = self.__get_indices()

    def __getitem__(self, index):
        return self.dataset.__getitem__(self.indices[index])

    def __len__(self):
        return len(self.indices)

    def __get_indices(self):
        indices = []
        for index, item in enumerate(self.dataset):
            if all(f(index, item) for f in self.filters):
                indices.append(index)
        return indices


class DatasetMapper:
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            mappers: Iterable[Callable]):
        self.dataset = dataset
        self.mappers = mappers

    def __getitem__(self, index):
        item = self.dataset.__getitem__(index)
        for mapper in self.mappers:
            item = mapper(index, item)
        return item

    def __len__(self):
        return self.dataset.__len__()


def sub_dataset(dataset: torch.utils.data.Dataset, indices: Iterable):
    r"""
    Subset of a dataset at specified indices in order.
    """
    indices = sorted(set(indices))
    return torch.utils.data.Subset(dataset, indices)


def sample_dataset(dataset: torch.utils.data.Dataset, index: int):
    return sub_dataset(dataset, [index])


def dataset_with_indices(dataset: torch.utils.data.Dataset):
    return DatasetMapper(dataset, [lambda index, item: (*item, index)])


def split_dataset(dataset: torchvision.datasets.VisionDataset) -> Generator:
    return (
        torch.utils.data.Subset(
            dataset,
            [index]) for index in range(
            len(dataset)))


class DatasetUtil:
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset
        self.__channel = None
        self.__len = None

    @property
    def len(self):
        if self.__len is None:
            self.__len = len(self.dataset)
        return self.__len

    @property
    def channel(self):
        if self.__channel is not None:
            return self.__channel
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1)
        channel = 0
        for x, _ in dataloader:
            channel = x.shape[1]
            break
        self.__channel = channel
        return self.__channel

    def get_mean_and_std(self):
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1)
        mean = torch.zeros(self.channel)
        for x, _ in dataloader:
            for i in range(self.channel):
                mean[i] += x[:, i, :, :].mean()
        mean.div_(self.len)

        wh = None
        std = torch.zeros(self.channel)
        for x, _ in dataloader:
            if wh is None:
                wh = x.shape[2] * x.shape[3]
            for i in range(self.channel):
                std[i] += torch.sum((x[:, i, :, :] -
                                     mean[i].data.item()) ** 2) / wh

        std = std.div(self.len).sqrt()
        return mean, std

    @staticmethod
    def get_labels_from_target(target):
        if isinstance(target, int):
            return set([target])
        if isinstance(target, torch.Tensor):
            return set([target.data.item()])
        if isinstance(target, dict):
            if "labels" in target:
                return set(target["labels"].tolist())
        raise RuntimeError("can't extract labels from target: " + str(target))

    @staticmethod
    def get_label_from_target(target):
        labels = DatasetUtil.get_labels_from_target(target)
        assert len(labels) == 1
        return next(iter(labels))

    def get_sample_label(self, index):
        return DatasetUtil.get_label_from_target(self.dataset[index][1])

    def get_labels(self) -> set:
        def count_instance(container, instance):
            labels = DatasetUtil.get_labels_from_target(instance[1])
            container.update(labels)
            return container

        return functools.reduce(count_instance, self.dataset, set())

    def split_by_label(self) -> dict:
        label_map: dict = {}
        for index, _ in enumerate(self.dataset):
            label = self.get_sample_label(index)
            if label not in label_map:
                label_map[label] = []
            label_map[label].append(index)
        for label, indices in label_map.items():
            label_map[label] = dict()
            label_map[label]["indices"] = indices
        return label_map

    def get_label_number(self) -> int:
        return len(self.get_labels())

    def save_sample_image(self, idx, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if isinstance(self.dataset[idx][0], PIL.Image.Image):
            self.dataset[idx][0].save(path)
            return
        torchvision.utils.save_image(self.dataset[idx][0], path)

    def split_by_ratio(self, parts: list, by_label: bool = True) -> list:
        assert parts
        sub_dataset_indices_list: list = []
        for _ in parts:
            sub_dataset_indices_list.append([])

        if by_label:
            for _, v in self.split_by_label().items():
                label_indices_list = sorted(v["indices"])
                for i, part in enumerate(parts):
                    delimiter = int(len(label_indices_list)
                                    * part / sum(parts[i:]))
                    sub_dataset_indices_list[i] += label_indices_list[:delimiter]
                    label_indices_list = label_indices_list[delimiter:]
        else:
            label_indices_list = list(range(len(self.dataset)))
            for i, part in enumerate(parts):
                delimiter = int(len(label_indices_list)
                                * part / sum(parts[i:]))
                sub_dataset_indices_list[i] += label_indices_list[:delimiter]
                label_indices_list = label_indices_list[delimiter:]

        return [sub_dataset(self.dataset, indices)
                for indices in sub_dataset_indices_list]

    def sample_subset(self, percentage: float) -> dict:
        label_map = self.split_by_label()
        sample_indices = dict()
        for label, v in label_map.items():
            sample_size = int(len(v["indices"]) * percentage)
            if sample_size == 0:
                get_logger().warning("percentage is too small, use sample size 1")
                sample_size = 1
            sample_indices[label] = random.sample(v["indices"], k=sample_size)
        return sample_indices

    def randomize_subset_label(self, percentage: float) -> dict:
        sample_indices = self.sample_subset(percentage)
        labels = sample_indices.keys()
        randomized_label_map = dict()
        for label, indices in sample_indices.items():
            other_labels = list(set(labels) - set([label]))
            for index in indices:
                randomized_label_map[index] = random.choice(other_labels)
                assert randomized_label_map[index] != self.dataset[index][1]
        return randomized_label_map


def replace_dataset_labels(dataset, label_map: dict):
    assert label_map

    def mapper(index, item):
        if index in label_map:
            assert label_map[index] != item[1]
            item = list(item)
            item[1] = label_map[index]
            return tuple(item)
        return item

    return DatasetMapper(dataset, [mapper])


__dataset_dir = os.path.join(os.path.expanduser("~"), "pytorch_dataset")


def set_dataset_dir(new_dataset_dir):
    global __dataset_dir
    __dataset_dir = new_dataset_dir


__datasets: dict = dict()


def get_dataset(name: str, phase: MachineLearningPhase):
    root_dir = os.path.join(__dataset_dir, name)
    for_training = phase in (
        MachineLearningPhase.Training,
        MachineLearningPhase.Validation,
    )
    training_dataset_parts = None
    by_label = True
    if name == "MNIST":
        dataset = torchvision.datasets.MNIST(
            root=root_dir,
            train=for_training,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.1307], std=[0.3081]),
                ]
            ),
        )
        training_dataset_parts = [5, 1]
    elif name == "FashionMNIST":
        transform = [
            transforms.Resize((32, 32)),
        ]
        if phase == MachineLearningPhase.Training:
            transform.append(transforms.RandomHorizontalFlip())
        transform += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2860], std=[0.3530]),
        ]
        dataset = torchvision.datasets.FashionMNIST(
            root=root_dir,
            train=for_training,
            download=True,
            transform=transforms.Compose(transform),
        )
        training_dataset_parts = [5, 1]
    elif name == "CIFAR10":
        transform = []

        if phase == MachineLearningPhase.Training:
            transform += [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]

        transform += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            ),
        ]
        dataset = torchvision.datasets.CIFAR10(
            root=root_dir,
            train=for_training,
            download=True,
            transform=transforms.Compose(transform),
        )
        training_dataset_parts = [4, 1]
    elif name == "SVHN":
        dataset = torchvision.datasets.SVHN(
            root=root_dir,
            split="extra",
            download=True,
            transform=transforms.ToTensor(),
        )
        mean, std = DatasetUtil(dataset).get_mean_and_std()

        transform = []
        if phase == MachineLearningPhase.Training:
            transform += [
                transforms.RandomCrop(32, padding=4),
            ]

        transform += [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        dataset = torchvision.datasets.SVHN(
            root=root_dir,
            split=("extra" if for_training else "test"),
            download=True,
            transform=transforms.Compose(transform),
        )
        training_dataset_parts = [4, 1]
    elif name == "WebankStreet":
        dataset = WebankStreetDataset(
            "/home/cyy/Street_Dataset/Street_Dataset",
            train=True,
            transform=transforms.ToTensor(),
        )
        mean, std = DatasetUtil(dataset).get_mean_and_std()
        transform = []
        if phase == MachineLearningPhase.Training:
            transform += [
                transforms.RandomHorizontalFlip(),
            ]

        transform += [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        dataset = WebankStreetDataset(
            "/home/cyy/Street_Dataset/Street_Dataset",
            train=for_training,
            transform=transforms.Compose(transform),
        )
        training_dataset_parts = [4, 1]
        by_label = False
    else:
        raise NotImplementedError(name)
    if not for_training:
        return dataset

    if name not in __datasets:
        dataset_util = DatasetUtil(dataset)
        training_dataset, validation_dataset = tuple(
            dataset_util.split_by_ratio(
                training_dataset_parts, by_label=by_label))
        __datasets[name] = dict()
        __datasets[name][MachineLearningPhase.Training] = training_dataset
        __datasets[name][MachineLearningPhase.Validation] = validation_dataset
    return __datasets[name][phase]


def get_dataset_label_names(name: str) -> List[str]:
    if name == "MNIST":
        return [str(a) for a in range(10)]
    if name == "FashionMNIST":
        return [
            "T-shirt",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]
    if name == "CIFAR10":
        return [
            "Airplane",
            "Automobile",
            "Bird",
            "Cat",
            "Deer",
            "Dog",
            "Frog",
            "Horse",
            "Ship",
            "Truck",
        ]
    raise NotImplementedError(name)


def dataset_append_transform(dataset, transform_fun):
    assert hasattr(dataset, "transform")
    dataset.transform = transforms.Compose([dataset.transform, transform_fun])
    return dataset
