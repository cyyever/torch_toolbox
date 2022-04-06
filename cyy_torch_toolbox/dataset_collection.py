import functools
import json
import os
import pickle
import threading
from typing import Callable, Dict, List

import torch
from cyy_naive_lib.log import get_logger
from ssd_checker import is_ssd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms

from cyy_torch_toolbox.dataset import (convert_iterable_dataset_to_map,
                                       replace_dataset_labels, sub_dataset)
from cyy_torch_toolbox.dataset_repository import get_dataset_constructors
from cyy_torch_toolbox.dataset_transformers.tokenizer import Tokenizer
from cyy_torch_toolbox.dataset_util import (  # CachedVisionDataset,
    DatasetSplitter, DatasetUtil, TextDatasetUtil, VisionDatasetUtil)
from cyy_torch_toolbox.ml_type import DatasetType, MachineLearningPhase
from cyy_torch_toolbox.reflection import get_kwarg_names


class DatasetCollection:
    def __init__(
        self,
        training_dataset: torch.utils.data.Dataset,
        validation_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        dataset_type: DatasetType,
        name: str,
    ):
        assert training_dataset is not None
        assert validation_dataset is not None
        assert test_dataset is not None
        self.__datasets: Dict[MachineLearningPhase, torch.utils.data.Dataset] = {}
        self.__datasets[MachineLearningPhase.Training] = training_dataset
        self.__datasets[MachineLearningPhase.Validation] = validation_dataset
        self.__datasets[MachineLearningPhase.Test] = test_dataset
        self.__dataset_type = dataset_type
        self.__transforms: dict = {}
        self.__target_transforms: dict = {}
        self.__input_batch_transforms: dict = {}
        for phase in (
            MachineLearningPhase.Training,
            MachineLearningPhase.Test,
            MachineLearningPhase.Validation,
        ):
            self.__transforms[phase] = []
            self.__target_transforms[phase] = []
            self.__input_batch_transforms[phase] = []
        self.__name = name
        self.__tokenizer: Tokenizer = None

    @property
    def tokenizer(self) -> Tokenizer:
        if self.__tokenizer is None:
            self.__tokenizer = Tokenizer(self)
        return self.__tokenizer

    @property
    def dataset_type(self):
        return self.__dataset_type

    def transform_dataset(
        self, phase: MachineLearningPhase, transformer: Callable
    ) -> None:
        dataset = self.get_dataset(phase)
        dataset_util = self.get_dataset_util(phase)
        self.__datasets[phase] = transformer(dataset, dataset_util)

    def foreach_dataset(self):
        for phase in (
            MachineLearningPhase.Training,
            MachineLearningPhase.Test,
            MachineLearningPhase.Validation,
        ):
            yield self.get_dataset(phase=phase)

    def transform_all_datasets(self, transformer: Callable) -> None:
        for phase in (
            MachineLearningPhase.Training,
            MachineLearningPhase.Test,
            MachineLearningPhase.Validation,
        ):
            self.transform_dataset(phase, transformer)

    def get_dataset(self, phase: MachineLearningPhase) -> torch.utils.data.Dataset:
        return self.__datasets[phase]

    def get_training_dataset(self) -> torch.utils.data.Dataset:
        return self.get_dataset(MachineLearningPhase.Training)

    def get_original_dataset(
        self, phase: MachineLearningPhase
    ) -> torch.utils.data.Dataset:
        dataset = self.get_dataset(phase)
        if hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        return dataset

    def get_dataset_util(
        self, phase: MachineLearningPhase = MachineLearningPhase.Test
    ) -> DatasetUtil:
        match self.dataset_type:
            case DatasetType.Vision:
                class_name = VisionDatasetUtil
            case DatasetType.Text:
                class_name = TextDatasetUtil
            case _:
                class_name = DatasetUtil
        return class_name(
            self.get_dataset(phase),
            transforms=self.__transforms[phase],
            target_transforms=self.__target_transforms[phase],
            name=self.name,
        )

    def append_transforms(self, transforms, phases=None):
        for k in MachineLearningPhase:
            if phases is not None and k not in phases:
                continue
            self.__transforms[k] += transforms

    def append_input_batch_transform(self, transform, phases=None):
        for k in MachineLearningPhase:
            if phases is not None and k not in phases:
                continue
            self.__input_batch_transforms[k].append(transform)

    def append_target_transform(self, transform, phases=None):
        for k in MachineLearningPhase:
            if phases is not None and k not in phases:
                continue
            self.__target_transforms[k].append(transform)

    def append_transform(self, transform, phases=None):
        return self.append_transforms([transform], phases)

    def prepend_transform(self, transform, phase=None):
        for k in MachineLearningPhase:
            if phase is not None and k != phase:
                continue
            self.__transforms[k].insert(0, transform)

    @property
    def name(self) -> str:
        return self.__name

    def get_labels(self, use_cache: bool = True) -> set:
        cache_dir = DatasetCollection.__get_dataset_cache_dir(self.name)
        pickle_file = os.path.join(cache_dir, "labels.pk")

        def computation_fun():
            if self.name.lower() == "imagenet":
                return range(1000)
            return self.get_dataset_util(
                phase=MachineLearningPhase.Training
            ).get_labels()

        if not use_cache:
            return computation_fun()

        return DatasetCollection.__get_cache_data(pickle_file, computation_fun)

    def collate_batch(self, batch, phase):
        inputs = []
        targets = []
        other_info = []
        transforms = self.__transforms[phase]
        input_batch_transforms = self.__input_batch_transforms[phase]
        target_transforms = self.__target_transforms[phase]
        for item in batch:
            if len(item) == 3:
                input, target, tmp = item
                other_info.append(tmp)
            else:
                input, target = item
            for f in transforms:
                input = f(input)
            inputs.append(input)
            for f in target_transforms:
                target = f(target)
            targets.append(target)
        if input_batch_transforms:
            for f in input_batch_transforms:
                inputs = f(inputs)
        else:
            inputs = default_collate(inputs)
        targets = default_collate(targets)
        if other_info:
            other_info = default_collate(other_info)
            return inputs, targets, other_info
        return inputs, targets

    def get_raw_data(self, phase: MachineLearningPhase, index: int) -> tuple:
        if self.dataset_type == DatasetType.Vision:
            dataset_util = self.get_dataset_util(phase)
            return (
                dataset_util.get_sample_image(index),
                dataset_util.get_sample_label(index),
            )
        if self.dataset_type == DatasetType.Text:
            dataset_util = self.get_dataset_util(phase)
            return (
                dataset_util.get_sample_text(index),
                dataset_util.get_sample_label(index),
            )
        raise RuntimeError("Unimplemented Code")

    def generate_raw_data(self, phase: MachineLearningPhase):
        if self.dataset_type == DatasetType.Vision:
            dataset_util = self.get_dataset_util(phase)
            return (
                (
                    dataset_util.get_sample_image(i),
                    dataset_util.get_sample_label(i),
                )
                for i in range(len(dataset_util.dataset))
            )
        raise RuntimeError("Unimplemented Code")

    def get_label_names(self) -> List[str]:
        cache_dir = DatasetCollection.__get_dataset_cache_dir(self.name)
        pickle_file = os.path.join(cache_dir, "label_names.pk")

        def computation_fun():
            label_names = self.get_dataset_util(
                phase=MachineLearningPhase.Training
            ).get_label_names()
            if not label_names:
                raise NotImplementedError(f"failed to get label names for {self.name}")
            return label_names

        return DatasetCollection.__get_cache_data(pickle_file, computation_fun)

    __dataset_root_dir: str = os.path.join(os.path.expanduser("~"), "pytorch_dataset")
    __lock = threading.RLock()

    @classmethod
    def get_dataset_root_dir(cls):
        with cls.__lock:
            return os.getenv("pytorch_dataset_root_dir", cls.__dataset_root_dir)

    @classmethod
    def set_dataset_root_dir(cls, root_dir: str):
        with cls.__lock:
            cls.__dataset_root_dir = root_dir

    @classmethod
    def __get_dataset_dir(cls, name: str):
        dataset_dir = os.path.join(cls.get_dataset_root_dir(), name)
        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir, exist_ok=True)
        if name.lower() == "imagenet":
            if not is_ssd(dataset_dir):
                get_logger().warning("dataset %s is not on a SSD disk", name)
        return dataset_dir

    @classmethod
    def __get_dataset_cache_dir(
        cls,
        name: str,
        phase: MachineLearningPhase = None,
    ) -> str:
        cache_dir = os.path.join(cls.__get_dataset_dir(name), ".cache")
        if phase is not None:
            cache_dir = os.path.join(cache_dir, str(phase))
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    @classmethod
    def get_by_name(cls, name: str, dataset_kwargs: dict = None):
        with cls.__lock:
            all_dataset_constructors = set()
            for dataset_type in DatasetType:
                dataset_constructor = get_dataset_constructors(dataset_type)
                if name in dataset_constructor:
                    return cls.__create_dataset_collection(
                        name, dataset_type, dataset_constructor[name], dataset_kwargs
                    )
                all_dataset_constructors |= dataset_constructor.keys()
            get_logger().error("supported datasets are %s", all_dataset_constructors)
            raise NotImplementedError(name)

    @classmethod
    def __get_mean_and_std(cls, name: str, dataset):
        cache_dir = cls.__get_dataset_cache_dir(name)
        pickle_file = os.path.join(cache_dir, "mean_and_std.pk")

        def computation_fun():
            return VisionDatasetUtil(dataset).get_mean_and_std()

        return cls.__get_cache_data(pickle_file, computation_fun)

    # @classmethod
    # def __get_dataset_size(cls, name: str):
    #     dataset_dir = cls.__get_dataset_dir(name)
    #     cache_dir = cls.__get_dataset_cache_dir(name)
    #     pickle_file = os.path.join(cache_dir, "dataset_size")

    #     def computation_fun():
    #         size = 0
    #         for path, _, files in os.walk(dataset_dir):
    #             for f in files:
    #                 size += os.path.getsize(os.path.join(path, f))
    #         return size

    #     return cls.__get_cache_data(pickle_file, computation_fun)

    @classmethod
    def __create_dataset_collection(
        cls,
        name: str,
        dataset_type: DatasetType,
        dataset_constructor,
        dataset_kwargs: dict = None,
    ):
        constructor_kwargs = get_kwarg_names(dataset_constructor)
        dataset_kwargs = cls.__prepare_dataset_kwargs(
            name, dataset_type, constructor_kwargs, dataset_kwargs
        )
        training_dataset = None
        validation_dataset = None
        test_dataset = None

        for phase in MachineLearningPhase:
            while True:
                try:
                    if "train" in constructor_kwargs:
                        # Some dataset only have train and test parts
                        if phase == MachineLearningPhase.Validation:
                            break
                        dataset_kwargs["train"] = phase == MachineLearningPhase.Training
                    if "split" in constructor_kwargs:
                        if phase == MachineLearningPhase.Training:
                            dataset_kwargs["split"] = "train"
                        elif phase == MachineLearningPhase.Validation:
                            if dataset_type == DatasetType.Text:
                                dataset_kwargs["split"] = "valid"
                            else:
                                dataset_kwargs["split"] = "val"
                        else:
                            dataset_kwargs["split"] = "test"
                    if "subset" in constructor_kwargs:
                        if phase == MachineLearningPhase.Training:
                            dataset_kwargs["subset"] = "training"
                        elif phase == MachineLearningPhase.Validation:
                            dataset_kwargs["subset"] = "validation"
                        else:
                            dataset_kwargs["subset"] = "testing"
                    dataset = dataset_constructor(**dataset_kwargs)
                    if name == "IMDB":
                        dataset = convert_iterable_dataset_to_map(
                            dataset, swap_item=True
                        )
                    if phase == MachineLearningPhase.Training:
                        training_dataset = dataset
                    elif phase == MachineLearningPhase.Validation:
                        validation_dataset = dataset
                    else:
                        test_dataset = dataset
                    break
                except Exception as e:
                    if "of splits is not supported for dataset" in str(e):
                        break
                    if "for argument split. Valid values are" in str(e):
                        break
                    # split = dataset_kwargs.get("split", None)
                    # # if split == "test":
                    # #     break
                    raise e

        # cache the datasets in memory to avoid IO and decoding
        # if cls.__get_dataset_size(name) / (1024 * 1024 * 1024) <= 1:
        #     get_logger().warning("cache dataset")
        #     training_dataset = CachedVisionDataset(training_dataset)
        #     if validation_dataset is not None:
        #         validation_dataset = CachedVisionDataset(validation_dataset)
        #     if test_dataset is not None:
        #         test_dataset = CachedVisionDataset(test_dataset)

        if validation_dataset is None or test_dataset is None:
            if validation_dataset is not None:
                splitted_dataset = validation_dataset
                get_logger().warning("split validation dataset for %s", name)
            else:
                splitted_dataset = test_dataset
                get_logger().warning("split test dataset for %s", name)
            (validation_dataset, test_dataset,) = cls.__split_for_validation(
                cls.__get_dataset_cache_dir(name), splitted_dataset
            )
        dc = cls(training_dataset, validation_dataset, test_dataset, dataset_type, name)

        try:
            get_logger().info("training_dataset len %s", len(training_dataset))
            get_logger().info("validation_dataset len %s", len(validation_dataset))
            get_logger().info("test_dataset len %s", len(test_dataset))
        except BaseException:
            pass

        if dataset_type == DatasetType.Vision:
            mean, std = cls.__get_mean_and_std(
                name, torch.utils.data.ConcatDataset(list(dc.__datasets.values()))
            )
            dc.append_transform(transforms.Normalize(mean=mean, std=std))
            if name not in ("SVHN", "MNIST"):
                dc.append_transform(
                    transforms.RandomHorizontalFlip(),
                    phases={MachineLearningPhase.Training},
                )
            # if name in ("CIFAR10", "CIFAR100"):
            #     dc.append_transform(
            #         # transforms.RandomCrop(32, padding=4),
            #         phases={MachineLearningPhase.Training},
            #     )
            if name.lower() == "imagenet":
                dc.append_transform(
                    transforms.RandomResizedCrop(224),
                    phases={MachineLearningPhase.Training},
                )
                dc.append_transforms(
                    [
                        transforms.RandomResizedCrop(224),
                        # transforms.Resize(256),
                        # transforms.CenterCrop(224),
                    ],
                    phases={MachineLearningPhase.Validation, MachineLearningPhase.Test},
                )
        if dataset_type == DatasetType.Text:
            dc.append_transform(dc.tokenizer)
            dc.append_transform(torch.tensor)
            label_names = None
            if isinstance(next(iter(dc.get_training_dataset()))[1], str):
                label_names = dc.get_label_names()
            if label_names is not None:
                dc.append_target_transform(
                    functools.partial(cls.get_label, label_names=label_names)
                )
                get_logger().warning("covert string label to int")
            if name == "IMDB":
                dc.append_input_batch_transform(
                    functools.partial(
                        pad_sequence, padding_value=dc.tokenizer.vocab["<pad>"]
                    )
                )

        return dc

    @classmethod
    def __prepare_dataset_kwargs(
        cls,
        name: str,
        dataset_type: DatasetType,
        constructor_kwargs: set,
        dataset_kwargs: dict = None,
    ) -> dict:
        if dataset_kwargs is None:
            dataset_kwargs = {}
        if "root" not in dataset_kwargs:
            dataset_kwargs["root"] = cls.__get_dataset_dir(name)
        if "download" not in dataset_kwargs:
            dataset_kwargs["download"] = True
        if dataset_type == DatasetType.Vision:
            if "transform" not in dataset_kwargs:
                dataset_kwargs["transform"] = transforms.Compose(
                    [transforms.ToTensor()]
                )

        discarded_dataset_kwargs = set()
        for k in dataset_kwargs:
            if k not in constructor_kwargs:
                discarded_dataset_kwargs.add(k)
        if discarded_dataset_kwargs:
            get_logger().warning(
                "discarded_dataset_kwargs %s", discarded_dataset_kwargs
            )
            for k in discarded_dataset_kwargs:
                dataset_kwargs.pop(k)
        return dataset_kwargs

    @staticmethod
    def __split_for_validation(cache_dir, splitted_dataset):
        pickle_file = os.path.join(cache_dir, "split_index_lists.pk")
        dataset_util = DatasetSplitter(splitted_dataset)
        split_index_lists = DatasetCollection.__read_data(pickle_file)
        if split_index_lists is not None:
            return dataset_util.split_by_indices(split_index_lists)
        datasets = dataset_util.iid_split([1, 1])
        DatasetCollection.__write_data(pickle_file, [d.indices for d in datasets])
        return datasets

    @staticmethod
    def __get_cache_data(path: str, computation_fun: Callable) -> dict:
        with DatasetCollection.__lock:
            data = DatasetCollection.__read_data(path)
            if data is not None:
                return data
            data = computation_fun()
            if data is None:
                raise RuntimeError("data is None")
            DatasetCollection.__write_data(path, data)
            return data

    @staticmethod
    def __read_data(path):
        if not os.path.isfile(path):
            return None
        fd = os.open(path, flags=os.O_RDONLY)
        with os.fdopen(fd, "rb") as f:
            res = pickle.load(f)
        return res

    @staticmethod
    def __write_data(path, data):
        fd = os.open(path, flags=os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def get_label(cls, label_name, label_names):
        reversed_label_names = {v: k for k, v in label_names.items()}
        return reversed_label_names[label_name]


class DatasetCollectionConfig:
    def __init__(self, dataset_name=None):
        self.dataset_name = dataset_name
        self.dataset_kwargs = {}
        self.training_dataset_percentage = None
        self.training_dataset_indices_path = None
        self.training_dataset_label_map_path = None
        self.training_dataset_label_map = None
        self.training_dataset_label_noise_percentage = None

    def add_args(self, parser):
        if self.dataset_name is None:
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
            with open(args.dataset_arg_json_path, "rt", encoding="utf-8") as f:
                self.dataset_kwargs = json.load(f)

    def create_dataset_collection(self, save_dir=None):
        if self.dataset_name is None:
            raise RuntimeError("dataset_name is None")

        dc = DatasetCollection.get_by_name(self.dataset_name, self.dataset_kwargs)

        dc.transform_dataset(
            MachineLearningPhase.Training,
            functools.partial(self.__transform_training_dataset, save_dir=save_dir),
        )
        return dc

    def __transform_training_dataset(
        self, training_dataset, dataset_util, save_dir=None
    ) -> torch.utils.data.Dataset:
        subset_indices = None
        if self.training_dataset_percentage is not None:
            subset_dict = dataset_util.iid_sample(self.training_dataset_percentage)
            subset_indices = sum(subset_dict.values(), [])
            with open(
                os.path.join(save_dir, "training_dataset_indices.json"),
                mode="wt",
                encoding="utf-8",
            ) as f:
                json.dump(subset_indices, f)

        if self.training_dataset_indices_path is not None:
            assert subset_indices is None
            get_logger().info(
                "use training_dataset_indices_path %s",
                self.training_dataset_indices_path,
            )
            with open(self.training_dataset_indices_path, "r", encoding="utf-8") as f:
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
                encoding="utf-8",
            ) as f:
                json.dump(label_map, f)

        if self.training_dataset_label_map_path is not None:
            assert label_map is not None
            get_logger().info(
                "use training_dataset_label_map_path %s",
                self.training_dataset_label_map_path,
            )
            with open(self.training_dataset_label_map_path, "r", encoding="utf-8") as f:
                self.training_dataset_label_map = json.load(f)

        if self.training_dataset_label_map is not None:
            training_dataset = replace_dataset_labels(
                training_dataset, self.training_dataset_label_map
            )
        return training_dataset
