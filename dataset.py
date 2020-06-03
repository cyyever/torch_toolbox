import functools
import os
import random

import torch
import torchvision
import torchvision.transforms as transforms


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


def dataset_exclude_samples(dataset, excluded_indices):
    return DatasetFilter(
        dataset, [
            lambda index, _: index not in excluded_indices])


def dataset_with_indices(dataset):
    return DatasetMapper(dataset, [lambda index, item: (*item, index)])


def split_dataset(dataset):
    return [
        torch.utils.data.Subset(
            dataset,
            [index]) for index in range(
            len(dataset))]


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


def radomize_subset_label(dataset, percentage):
    class_map = split_dataset_by_class(dataset)
    labels = set(class_map.keys())
    for label, v in class_map.items():
        other_labels = labels - set([label])
        print(label, other_labels)
        indices = v["indices"]
        radomized_subset_size = int(len(v["dataset"]) * percentage)
        randomized_indices = random.sample(indices, k=radomized_subset_size)
        randomized_label_map = dict()
        for index in randomized_indices:
            randomized_label_map[index] = random.choice(other_labels)
        print(randomized_label_map)
    return randomized_label_map


def replace_dataset_labels(dataset, label_map):
    def mapper(index, item):
        if index in label_map:
            item[1] = label_map[index]

    return DatasetMapper(dataset, [mapper])


def get_classes(dataset):
    def count_instance(container, instance):
        label = instance[1]
        container.add(label)
        return container

    return functools.reduce(count_instance, dataset, set())


def get_class_count(dataset):
    def count_instance(container, instance):
        label = instance[1]
        if isinstance(label, torch.Tensor):
            label = label.data.item()
        container[label] = container.get(label, 0) + 1
        return container

    return functools.reduce(count_instance, dataset, dict())


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
    if for_train:
        root_dir = os.path.join(root_dir, "training")
    else:
        root_dir = os.path.join(root_dir, "validation")
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
        return torchvision.datasets.FashionMNIST(
            root=root_dir,
            train=for_train,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.2860], std=[0.3530]),
                ]
            ),
        )
    if name == "STL10":
        root_dir = os.path.dirname(root_dir)
        if for_train:
            split = "train"
        else:
            split = "test"

        return torchvision.datasets.STL10(
            root=root_dir,
            split=split,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(
                        96,
                        padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[
                            0.4467,
                            0.4398,
                            0.4066],
                        std=[
                            0.2603,
                            0.2566,
                            0.2713]),
                ]),
        )
    if name == "CIFAR10":
        return torchvision.datasets.CIFAR10(
            root=root_dir,
            train=for_train,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(
                        32,
                        padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[
                            0.4914,
                            0.4822,
                            0.4465],
                        std=[
                            0.2470,
                            0.2435,
                            0.2616]),
                ]),
        )
    raise NotImplementedError(name)
