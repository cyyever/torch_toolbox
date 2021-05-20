import copy
import inspect
import json
import os
import pickle
from typing import Callable, Dict, List

import torch
import torchaudio
import torchvision
import torchvision.transforms as transforms
from cyy_naive_lib.log import get_logger

import audio_datasets as local_audio_datasets
import vision_datasets as local_vision_datasets
from dataset import (DatasetToMelSpectrogram, DatasetUtil,
                     replace_dataset_labels, sub_dataset)
from hyper_parameter import HyperParameter
from ml_type import DatasetType, MachineLearningPhase


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
        self.__origin_datasets: Dict[
            MachineLearningPhase, torch.utils.data.Dataset
        ] = dict()
        for k, v in self.__datasets.items():
            self.__origin_datasets[k] = v
        self.__name = name

    def set_origin_dataset(self, phase: MachineLearningPhase, dataset):
        self.__origin_datasets[phase] = dataset

    def transform_dataset(self, phase: MachineLearningPhase, transformer: Callable):
        dataset = self.get_dataset(phase)
        self.__datasets[phase] = transformer(dataset)

    def get_training_dataset(self) -> torch.utils.data.Dataset:
        return self.get_dataset(MachineLearningPhase.Training)

    def get_dataset(self, phase: MachineLearningPhase) -> torch.utils.data.Dataset:
        assert phase in self.__datasets
        return self.__datasets[phase]

    def get_dataset_util(
        self, phase: MachineLearningPhase = MachineLearningPhase.Test
    ) -> DatasetUtil:
        return DatasetUtil(self.get_dataset(phase))

    def append_transform(self, transform, phase=None):
        origin_datasets = set()
        for k in MachineLearningPhase:
            if phase is not None and k != phase:
                continue
            origin_datasets.add(self.__origin_datasets[k])
        for dataset in origin_datasets:
            DatasetUtil(dataset).append_transform(transform)

    def prepend_transform(self, transform, phase=None):
        origin_datasets = set()
        for k in MachineLearningPhase:
            if phase is not None and k != phase:
                continue
            origin_datasets.add(self.__origin_datasets[k])
        for dataset in origin_datasets:
            DatasetUtil(dataset).prepend_transform(transform)

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

    def get_labels(self) -> set:
        cache_dir = DatasetCollection.get_dataset_cache_dir(self.name)
        pickle_file = os.path.join(cache_dir, "labels.pk")
        if os.path.isfile(pickle_file):
            return pickle.load(open(pickle_file, "rb"))
        with open(pickle_file, "wb") as f:
            labels = self.get_dataset_util().get_labels()
            pickle.dump(labels, f)
            return labels

    def get_label_names(self) -> List[str]:
        if hasattr(self.get_training_dataset(), "classes"):
            return getattr(self.get_training_dataset(), "classes")

        vision_dataset_cls = DatasetCollection.get_dataset_cls(DatasetType.Vision)
        if self.name not in vision_dataset_cls:
            get_logger().error("supported datasets are %s", vision_dataset_cls.keys())
            raise NotImplementedError(self.name)
        vision_dataset_cls = vision_dataset_cls[self.name]
        if hasattr(vision_dataset_cls, "classes"):
            return getattr(vision_dataset_cls, "classes")
        get_logger().error("%s has no classes", self.name)
        raise NotImplementedError(self.name)

    __dataset_root_dir: str = os.path.join(os.path.expanduser("~"), "pytorch_dataset")
    __dataset_collections: Dict = dict()

    @staticmethod
    def set_dataset_root_dir(root_dir: str):
        DatasetCollection.__dataset_root_dir = root_dir

    @staticmethod
    def get_dataset_dir(name: str):
        dataset_dir = os.path.join(DatasetCollection.__dataset_root_dir, name)
        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir, exist_ok=True)
        return dataset_dir

    @staticmethod
    def get_dataset_cache_dir(name: str):
        cache_dir = os.path.join(DatasetCollection.get_dataset_dir(name), ".cache")
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    @staticmethod
    def get_dataset_cls(dataset_type: DatasetType):
        repositories = []
        if dataset_type == DatasetType.Vision:
            repositories = [torchvision.datasets, local_vision_datasets]
        elif dataset_type == DatasetType.Audio:
            repositories = [torchaudio.datasets, local_audio_datasets]
        datasets = dict()
        for repository in repositories:
            for name in dir(repository):
                dataset_class = getattr(repository, name)
                if not inspect.isclass(dataset_class):
                    continue
                if issubclass(dataset_class, torch.utils.data.Dataset):
                    datasets[name] = dataset_class
        return datasets

    @staticmethod
    def get_by_name(name: str, **kwargs):
        if name in DatasetCollection.__dataset_collections:
            return DatasetCollection.__dataset_collections[name]

        all_dataset_cls = set()
        for dataset_type in DatasetType:
            dataset_cls = DatasetCollection.get_dataset_cls(dataset_type)
            if name in dataset_cls:
                return DatasetCollection.__create_dataset_collection(
                    name, dataset_type, dataset_cls[name], **kwargs
                )
            all_dataset_cls |= dataset_cls.keys()
        get_logger().error("supported datasets are %s", all_dataset_cls)
        raise NotImplementedError(name)

    @staticmethod
    def __create_dataset_collection(
        name: str, dataset_type: DatasetType, dataset_cls, **kwargs
    ):
        root_dir = DatasetCollection.get_dataset_dir(name)
        training_dataset = None
        validation_dataset = None
        test_dataset = None
        dataset_kwargs = {
            "root": root_dir,
            "download": True,
        }
        vision_transform = transforms.Compose([transforms.ToTensor()])
        if dataset_type == DatasetType.Vision:
            dataset_kwargs["transform"] = vision_transform

        use_mel_spectrogram = kwargs.pop("use_mel_spectrogram", False)
        for k, v in kwargs.items():
            if k in dataset_kwargs:
                raise RuntimeError("key " + k + " is set by the library")
            dataset_kwargs[k] = v
        sig = inspect.signature(dataset_cls)

        for phase in MachineLearningPhase:
            while True:
                try:
                    if "train" in sig.parameters:
                        # Some dataset only have train and test parts
                        if phase == MachineLearningPhase.Validation:
                            break
                        dataset_kwargs["train"] = phase == MachineLearningPhase.Training
                    if "split" in sig.parameters:
                        if phase == MachineLearningPhase.Training:
                            dataset_kwargs["split"] = "train"
                        elif phase == MachineLearningPhase.Validation:
                            dataset_kwargs["split"] = "val"
                        else:
                            dataset_kwargs["split"] = "test"
                    if "subset" in sig.parameters:
                        if phase == MachineLearningPhase.Training:
                            dataset_kwargs["subset"] = "training"
                        elif phase == MachineLearningPhase.Validation:
                            dataset_kwargs["subset"] = "validation"
                        else:
                            dataset_kwargs["subset"] = "testing"
                    dataset = dataset_cls(**dataset_kwargs)
                    if phase == MachineLearningPhase.Training:
                        training_dataset = dataset
                    elif phase == MachineLearningPhase.Validation:
                        validation_dataset = dataset
                    else:
                        test_dataset = dataset
                    break
                except Exception as e:
                    raise e
                    split = dataset_kwargs.get("split", None)
                    if split is None:
                        raise e
                    if phase == MachineLearningPhase.Training:
                        raise e
                    # no validation dataset or test dataset
                    break

        cache_dir = DatasetCollection.get_dataset_cache_dir(name)

        splited_dataset = None
        if validation_dataset is None or test_dataset is None:
            if validation_dataset is not None:
                splited_dataset = validation_dataset
                get_logger().warning("split validation dataset for %s", name)
            else:
                splited_dataset = test_dataset
                get_logger().warning("split test dataset for %s", name)
            validation_dataset, test_dataset = DatasetCollection.__split_for_validation(
                cache_dir, splited_dataset
            )
        dc = DatasetCollection(training_dataset, validation_dataset, test_dataset, name)
        if splited_dataset is not None:
            dc.set_origin_dataset(MachineLearningPhase.Validation, splited_dataset)
            dc.set_origin_dataset(MachineLearningPhase.Test, splited_dataset)
        if dataset_type == DatasetType.Audio and use_mel_spectrogram:
            cache_dir = DatasetCollection.get_dataset_cache_dir(
                name + "_mel_spectrogram"
            )
            for phase in MachineLearningPhase:
                dc.transform_dataset(
                    phase,
                    lambda dataset: DatasetToMelSpectrogram(
                        copy.deepcopy(dataset), root=cache_dir
                    ),
                )
                dc.set_origin_dataset(phase, dc.get_dataset(phase=phase))
            dc.prepend_transform(vision_transform)
            dataset_type = DatasetType.Vision

        get_logger().info("training_dataset len %s", len(training_dataset))
        get_logger().info("validation_dataset len %s", len(validation_dataset))
        get_logger().info("test_dataset len %s", len(test_dataset))

        if dataset_type == DatasetType.Vision:
            pickle_file = os.path.join(cache_dir, "mean_and_std.pk")
            if os.path.isfile(pickle_file):
                mean, std = pickle.load(open(pickle_file, "rb"))
            else:
                total_dataset = torch.utils.data.ConcatDataset(
                    list(dc.__datasets.values())
                )
                mean, std = DatasetUtil(total_dataset).get_mean_and_std()
                with open(pickle_file, "wb") as f:
                    pickle.dump((mean, std), f)
            dc.append_transform(transforms.Normalize(mean=mean, std=std))
            if name not in ("SVHN", "MNIST"):
                dc.append_transform(
                    transforms.RandomHorizontalFlip(),
                    phase=MachineLearningPhase.Training,
                )
            if name in ("CIFAR10", "CIFAR100"):
                dc.append_transform(
                    transforms.RandomCrop(32, padding=4),
                    phase=MachineLearningPhase.Training,
                )
        if dataset_type == DatasetType.Audio:
            if name == "SPEECHCOMMANDS_SIMPLIFIED":
                dc.append_transform(
                    lambda tensor: torch.nn.ConstantPad1d(
                        (0, 16000 - tensor.shape[-1]), 0
                    )(tensor)
                )
        DatasetCollection.__dataset_collections[name] = dc
        return dc

    @staticmethod
    def __split_for_validation(cache_dir, splited_dataset):
        pickle_file = os.path.join(cache_dir, "split_index_lists.pk")
        if os.path.isfile(pickle_file):
            split_index_lists = pickle.load(open(pickle_file, "rb"))
            return tuple(
                sub_dataset(splited_dataset, indices) for indices in split_index_lists
            )
        with open(pickle_file, "wb") as f:
            dataset_util = DatasetUtil(splited_dataset)
            datasets = dataset_util.iid_split([1, 1])
            pickle.dump([d.indices for d in datasets], f)
            return datasets


class DatasetCollectionConfig:
    def __init__(self, dataset_name=None):
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
        parser.add_argument("--dataset_arg_json_path", type=str, default=None)

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
        if args.dataset_arg_json_path is not None:
            with open(args.dataset_arg_json_path, "rt") as f:
                self.dataset_args = json.load(f)

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
        self, training_dataset, save_dir=None
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
