# import multiprocessing
import json
import os
from typing import Callable, Dict, List

import torch
import torchvision
import torchvision.transforms as transforms
from cyy_naive_lib.log import get_logger

from dataset import DatasetUtil, replace_dataset_labels, sub_dataset
from hyper_parameter import HyperParameter
from ml_type import MachineLearningPhase


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
        return self.__datasets[phase]

    def get_dataloader(
        self,
        phase: MachineLearningPhase,
        hyper_parameter: HyperParameter,
    ):
        dataset = self.get_dataset(phase)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=hyper_parameter.batch_size,
            shuffle=(phase == MachineLearningPhase.Training),
        )

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
    def get_by_name(name: str, **kwargs):
        if name in DatasetCollection.__dataset_collections:
            return DatasetCollection.__dataset_collections[name]
        root_dir = DatasetCollection.get_dataset_dir(name)
        # for_training = phase in (MachineLearningPhase.Training,)
        training_dataset = None
        validation_dataset = None
        test_dataset = None
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
                to_grayscale = kwargs.get("to_grayscale", False)
                if to_grayscale:
                    get_logger().warning("convert %s to grayscale", name)
                    transform += [transforms.Grayscale()]
                if for_training:
                    transform += [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                    ]

                transform.append(transforms.ToTensor())
                # use MNIST mean and std
                if to_grayscale:
                    transform.append(transforms.Normalize(mean=[0.1307], std=[0.3081]))
                else:
                    transform.append(
                        transforms.Normalize(
                            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                        )
                    )
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
        else:
            raise NotImplementedError(name)

        if validation_dataset is None:
            dataset_util = DatasetUtil(test_dataset)
            test_dataset_parts = [1, 1]
            validation_dataset, test_dataset = tuple(
                dataset_util.iid_split(test_dataset_parts)
            )
        dc = DatasetCollection(training_dataset, validation_dataset, test_dataset, name)
        DatasetCollection.__dataset_collections[name] = dc
        return dc


class DatasetCollectionConfig:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset_args = dict()
        self.training_dataset_percentage = None
        self.training_dataset_indices_path = None
        self.training_dataset_label_map_path = None
        self.training_dataset_label_map = None
        self.training_dataset_label_noise_percentage = None

    def add_args(self, parser):
        parser.add_argument("--dataset_name", type=str, required=True)
        parser.add_argument("--training_dataset_percentage", type=float, default=None)
        parser.add_argument("--training_dataset_indices_path", type=str, default=None)
        parser.add_argument(
            "--training_dataset_label_noise_percentage", type=float, default=None
        )

    def load_args(self, args):
        for attr in dir(args):
            if attr.startswith("_"):
                continue
            if not hasattr(self, attr):
                continue
            get_logger().debug("set dataset collection config attr %s", attr)
            value = getattr(args, attr)
            if value is not None:
                setattr(self, attr, value)

    def create_dataset_collection(self, save_dir):
        if self.dataset_name is None:
            raise RuntimeError("dataset_name is None")

        dc = DatasetCollection.get_by_name(self.dataset_name, **self.dataset_args)
        dc.transform_dataset(
            MachineLearningPhase.Training,
            lambda dataset: self.__transform_training_dataset(dataset, save_dir),
        )
        return dc

    def __transform_training_dataset(
        self, training_dataset, save_dir
    ) -> torch.utils.data.Dataset:
        subset_indices = None
        if self.training_dataset_percentage is not None:
            subset_dict = DatasetUtil(training_dataset).iid_sample(
                self.training_dataset_percentage
            )
            subset_indices = sum(subset_dict.values(), [])
            with open(
                os.path.join(save_dir, "training_dataset_indices.json"),
                mode="wt",
            ) as f:
                json.dump(subset_indices, f)

        if self.training_dataset_indices_path is not None:
            assert subset_indices is None
            get_logger().info(
                "use training_dataset_indices_path %s",
                self.training_dataset_indices_path,
            )
            with open(self.training_dataset_indices_path, "r") as f:
                subset_indices = json.load(f)
        if subset_indices is not None:
            training_dataset = sub_dataset(training_dataset, subset_indices)

        label_map = None
        if self.training_dataset_label_noise_percentage:
            label_map = DatasetUtil(training_dataset).randomize_subset_label(
                self.training_dataset_label_noise_percentage
            )
            with open(
                os.path.join(
                    save_dir,
                    "training_dataset_label_map.json",
                ),
                mode="wt",
            ) as f:
                json.dump(label_map, f)

        if self.training_dataset_label_map_path is not None:
            assert label_map is not None
            get_logger().info(
                "use training_dataset_label_map_path %s",
                self.training_dataset_label_map_path,
            )
            self.training_dataset_label_map = json.load(
                open(self.training_dataset_label_map_path, "r")
            )

        if self.training_dataset_label_map is not None:
            training_dataset = replace_dataset_labels(
                training_dataset, self.training_dataset_label_map_path
            )
        return training_dataset
