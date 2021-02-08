import multiprocessing
import os
from typing import Callable, Dict, List

import torch
import torchvision
import torchvision.transforms as transforms

from dataset import DatasetUtil, dataset_with_indices
from hyper_parameter import HyperParameter
from ml_types import MachineLearningPhase


class DatasetCollection:
    def __init__(
        self,
        training_dataset: torch.utils.data.Dataset,
        validation_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        name=None,
    ):
        assert training_dataset is not None
        assert validation_dataset is not None
        assert test_dataset is not None
        self.__datasets: Dict[MachineLearningPhase, torch.utils.data.Dataset] = dict()
        self.__datasets[MachineLearningPhase.Training] = training_dataset
        self.__datasets[MachineLearningPhase.Validation] = validation_dataset
        self.__datasets[MachineLearningPhase.Test] = test_dataset
        self.__name = name

    def transform_dataset(self, phase: MachineLearningPhase, transformer: Callable):
        dataset = self.get_dataset(phase)
        self.__datasets[phase] = transformer(dataset)

    def get_dataset(self, phase: MachineLearningPhase) -> torch.utils.data.Dataset:
        assert phase in self.__datasets
        return self.__datasets.get(phase)

    def get_dataloader(
        self,
        phase: MachineLearningPhase,
        hyper_parameter: HyperParameter,
        with_indices=False,
    ):
        dataset = self.get_dataset(phase)
        if with_indices:
            dataset = dataset_with_indices(dataset)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=hyper_parameter.batch_size,
            shuffle=(phase == MachineLearningPhase.Training),
        )
        # num_workers=multiprocessing.cpu_count(),

        # collate_fn=hyper_parameter.__collate_fn,

    @property
    def name(self):
        return self.__name

    def get_label_names(self) -> List[str]:
        if self.name == "MNIST":
            return [str(a) for a in range(10)]
        if self.name == "FashionMNIST":
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
        if self.name == "CIFAR10":
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
        raise NotImplementedError(self.name)

    __dataset_root_dir: str = os.path.join(os.path.expanduser("~"), "pytorch_dataset")
    __dataset_collections: Dict = dict()

    @staticmethod
    def set_dataset_root_dir(root_dir: str):
        DatasetCollection.__dataset_root_dir = root_dir

    @staticmethod
    def get_dataset_dir(name: str):
        return os.path.join(DatasetCollection.__dataset_root_dir, name)

    @staticmethod
    def get_by_name(name: str):
        if name in DatasetCollection.__dataset_collections:
            return DatasetCollection.__dataset_collections[name]
        root_dir = DatasetCollection.get_dataset_dir(name)
        # for_training = phase in (MachineLearningPhase.Training,)
        training_dataset = None
        validation_dataset = None
        test_dataset = None
        by_label = True
        if name == "MNIST":
            for for_training in (True, False):
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
                if for_training:
                    training_dataset = dataset
                else:
                    test_dataset = dataset
        elif name == "FashionMNIST":
            for for_training in (True, False):
                transform = [
                    transforms.Resize((32, 32)),
                ]
                if for_training:
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
                if for_training:
                    training_dataset = dataset
                else:
                    test_dataset = dataset
        elif name == "CIFAR10":
            for for_training in (True, False):
                transform = []
                if for_training:
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
                if for_training:
                    training_dataset = dataset
                else:
                    test_dataset = dataset
        elif name == "CIFAR100":
            for for_training in (True, False):
                transform = []
                if for_training:
                    transform += [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                    ]

                transform += [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]
                    ),
                ]
                dataset = torchvision.datasets.CIFAR100(
                    root=root_dir,
                    train=for_training,
                    download=True,
                    transform=transforms.Compose(transform),
                )
                if for_training:
                    training_dataset = dataset
                else:
                    test_dataset = dataset
        elif name == "SVHN":
            dataset = torchvision.datasets.SVHN(
                root=root_dir,
                split="extra",
                download=True,
                transform=transforms.ToTensor(),
            )
            mean, std = DatasetUtil(dataset).get_mean_and_std()

            for for_training in (True, False):
                transform = []
                if for_training:
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
                if for_training:
                    training_dataset = dataset
                else:
                    test_dataset = dataset
        # elif name == "WebankStreet":
        #     dataset = WebankStreetDataset(
        #         "/home/cyy/Street_Dataset/Street_Dataset",
        #         train=True,
        #         transform=transforms.ToTensor(),
        #     )
        #     mean, std = DatasetUtil(dataset).get_mean_and_std()
        #     transform = []
        #     if phase == MachineLearningPhase.Training:
        #         transform += [
        #             transforms.RandomHorizontalFlip(),
        #         ]

        #     transform += [
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=mean, std=std),
        #     ]
        #     dataset = WebankStreetDataset(
        #         "/home/cyy/Street_Dataset/Street_Dataset",
        #         train=for_training,
        #         transform=transforms.Compose(transform),
        #     )
        #     training_dataset_parts = [4, 1]
        #     by_label = False
        else:
            raise NotImplementedError(name)

        if validation_dataset is None:
            dataset_util = DatasetUtil(test_dataset)
            test_dataset_parts = [1, 1]
            validation_dataset, test_dataset = tuple(
                dataset_util.split_by_ratio(test_dataset_parts, by_label=by_label)
            )
        dc = DatasetCollection(training_dataset, validation_dataset, test_dataset, name)
        DatasetCollection.__dataset_collections[name] = dc
        return dc


def dataset_append_transform(dataset, transform_fun):
    assert hasattr(dataset, "transform")
    dataset.transform = transforms.Compose([dataset.transform, transform_fun])
    return dataset
