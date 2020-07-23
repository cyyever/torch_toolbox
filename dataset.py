import functools
import os
import random
import PIL

import torch
import torchvision
import torchvision.transforms as transforms

from cyy_naive_lib.log import get_logger


class DatasetFilter:
    def __init__(self, dataset, filters):
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
    def __init__(self, dataset, mappers):
        self.dataset = dataset
        self.mappers = mappers

    def __getitem__(self, index):
        item = self.dataset.__getitem__(index)
        for mapper in self.mappers:
            item = mapper(index, item)
        return item

    def __len__(self):
        return self.dataset.__len__()


def sub_dataset(dataset, indices):
    r"""
    Subset of a dataset at specified indices in order.
    """
    indices = sorted(set(indices))
    return torch.utils.data.Subset(dataset, indices)


def complement_dataset(dataset, indices):
    return sub_dataset(dataset, set(range(len(dataset)) - set(indices)))


def dataset_with_indices(dataset):
    return DatasetMapper(dataset, [lambda index, item: (*item, index)])


def split_dataset(dataset):
    return (
        torch.utils.data.Subset(
            dataset,
            [index]) for index in range(
            len(dataset)))


def split_dataset_by_class(dataset):
    class_map = {}
    for index, sampler in enumerate(dataset):
        label = sampler[1]
        if isinstance(label, torch.Tensor):
            label = label.data.item()
        if label not in class_map:
            class_map[label] = []
        class_map[label].append(index)
    for label, indices in class_map.items():
        class_map[label] = dict()
        class_map[label]["indices"] = indices
        class_map[label]["dataset"] = torch.utils.data.Subset(dataset, indices)
    return class_map


def sample_subset(dataset, percentage):
    class_map = split_dataset_by_class(dataset)
    sample_indices = dict()
    for label, v in class_map.items():
        sample_size = int(len(v["indices"]) * percentage)
        if sample_size == 0:
            get_logger().warning("percentage is too small, use sample size 1")
            sample_size = 1
        sample_indices[label] = random.sample(v["indices"], k=sample_size)
    return sample_indices


def randomize_subset_label(dataset, percentage):
    sample_indices = sample_subset(dataset, percentage)
    labels = sample_indices.keys()
    randomized_label_map = dict()
    for label, indices in sample_indices.items():
        other_labels = list(set(labels) - set([label]))
        for index in indices:
            randomized_label_map[index] = random.choice(other_labels)
            assert randomized_label_map[index] != dataset[index][1]
    return randomized_label_map


def replace_dataset_labels(dataset, label_map: dict):
    assert label_map

    def mapper(index, item):
        if index in label_map:
            assert label_map[index] != item[1]
            return (item[0], label_map[index])
        return item

    return DatasetMapper(dataset, [mapper])


def get_classes(dataset):
    def count_instance(container, instance):
        label = instance[1]
        container.add(label)
        return container

    return functools.reduce(count_instance, dataset, set())


def get_class_count(dataset):
    res = dict()
    for k, v in split_dataset_by_class(dataset).items():
        res[k] = len(v["indices"])
    return res


def save_sample(dataset, idx, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(dataset[idx][0], PIL.Image.Image):
        dataset[idx][0].save(path)
        return
    torchvision.utils.save_image(dataset[idx][0], path)


def get_mean_and_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    mean = None
    channel = None
    for x, _ in dataloader:
        channel = x.shape[1]
        if mean is None:
            mean = torch.zeros(channel)
        for i in range(channel):
            mean[i] += x[:, i, :, :].mean()
    mean.div_(len(dataset))

    wh = None
    std = torch.zeros(channel)
    for x, _ in dataloader:
        if wh is None:
            wh = x.shape[2] * x.shape[3]
        for i in range(channel):
            std[i] += torch.sum((x[:, i, :, :] -
                                 mean[i].data.item()) ** 2) / wh

    std = std.div(len(dataset)).sqrt()
    return mean, std


__dataset_dir = os.path.join(os.path.expanduser("~"), "pytorch_dataset")


def set_dataset_dir(new_dataset_dir):
    global __dataset_dir
    __dataset_dir = new_dataset_dir


def get_dataset(name, for_train):
    root_dir = os.path.join(__dataset_dir, name)
    if name == "MNIST":
        return torchvision.datasets.MNIST(
            root=root_dir,
            train=for_train,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.1307], std=[0.3081]),
                ]
            ),
        )
    if name == "FashionMNIST":
        transform = [
            transforms.Resize((32, 32)),
        ]
        if for_train:
            transform.append(transforms.RandomHorizontalFlip())
        transform += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2860], std=[0.3530]),
        ]
        return torchvision.datasets.FashionMNIST(
            root=root_dir,
            train=for_train,
            download=True,
            transform=transforms.Compose(transform),
        )
    if name == "STL10":
        root_dir = os.path.dirname(root_dir)
        if for_train:
            split = "train"
        else:
            split = "test"

        transform = []
        if for_train:
            transform += [
                transforms.RandomCrop(96, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        transform += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713]
            ),
        ]
        return torchvision.datasets.STL10(
            root=root_dir,
            split=split,
            download=True,
            transform=transforms.Compose(transform),
        )
    if name == "CIFAR10":
        transform = []

        if for_train:
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
        return torchvision.datasets.CIFAR10(
            root=root_dir,
            train=for_train,
            download=True,
            transform=transforms.Compose(transform),
        )
    raise NotImplementedError(name)


def get_dataset_labels(name):
    if name == "MNIST":
        return list(range(10))
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
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
    raise NotImplementedError(name)
