import inspect
import json
import os
import pickle
import threading
from typing import Callable, Dict, List

import torch

try:
    import torchaudio

    has_torchaudio = True
except ModuleNotFoundError:
    has_torchaudio = False
import torchtext
import torchvision
from cyy_naive_lib.log import get_logger
from torchvision import transforms

if has_torchaudio:
    import datasets.audio as local_audio_datasets

import datasets.vision as local_vision_datasets
from dataset import (DatasetUtil, convert_iterable_dataset_to_map,
                     replace_dataset_labels, sub_dataset)
from ml_type import DatasetType, MachineLearningPhase
from pipelines.text_pipeline import TokenizerAndVocab


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
        self.__name = name
        self.__tokenizer_and_vocab = None

    @property
    def tokenizer_and_vocab(self):
        if self.__tokenizer_and_vocab is None:
            self.__tokenizer_and_vocab = TokenizerAndVocab(self.get_training_dataset())
        return self.__tokenizer_and_vocab

    @property
    def dataset_type(self):
        return self.__dataset_type

    def transform_dataset(self, phase: MachineLearningPhase, transformer: Callable):
        dataset = self.get_dataset(phase)
        self.__datasets[phase] = transformer(dataset)

    def transform_dataset_to_subset(
        self, phase: MachineLearningPhase, labels: set
    ) -> None:
        label_indices = self.__get_label_indices(phase)
        all_labels = self.get_label_names()
        if not labels.issubset(all_labels):
            get_logger().error(
                "[%s] is not a subset of [%s]", " ".join(labels), " ".join(all_labels)
            )
            raise RuntimeError("invalid dataset labels")
        total_indices = []
        for label_index, indices in label_indices.items():
            if all_labels[label_index] in labels:
                total_indices += indices["indices"]

        self.transform_dataset(
            phase, lambda dataset: sub_dataset(dataset, total_indices)
        )

    def get_training_dataset(self) -> torch.utils.data.Dataset:
        return self.get_dataset(MachineLearningPhase.Training)

    def get_dataset(self, phase: MachineLearningPhase) -> torch.utils.data.Dataset:
        assert phase in self.__datasets
        return self.__datasets[phase]

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
        return DatasetUtil(self.get_dataset(phase))

    def append_transforms(self, transforms, phases=None):
        origin_datasets = set()
        for k in MachineLearningPhase:
            if phases is not None and k not in phases:
                continue
            origin_datasets.add(self.get_original_dataset(k))
        for dataset in origin_datasets:
            for t in transforms:
                DatasetUtil(dataset).append_transform(t)

    def append_transform(self, transform, phases=None):
        return self.append_transforms([transform], phases)

    def prepend_transform(self, transform, phase=None):
        origin_datasets = set()
        for k in MachineLearningPhase:
            if phase is not None and k != phase:
                continue
            origin_datasets.add(self.get_original_dataset(k))
        for dataset in origin_datasets:
            DatasetUtil(dataset).prepend_transform(transform)

    @property
    def name(self) -> str:
        return self.__name

    def get_labels(self, use_cache: bool = True) -> set:
        cache_dir = DatasetCollection.__get_dataset_cache_dir(self.name)
        pickle_file = os.path.join(cache_dir, "labels.pk")

        def computation_fun():
            if self.name.lower() == "imagenet":
                return range(1000)
            training_dataset = self.get_training_dataset()
            if hasattr(training_dataset, "targets"):
                return self.get_dataset_util(
                    phase=MachineLearningPhase.Training
                ).get_labels()
            return self.get_dataset_util(phase=MachineLearningPhase.Test).get_labels()

        if not use_cache:
            return computation_fun()

        return DatasetCollection.__get_cache_data(pickle_file, computation_fun)

    def get_collate_fn(self):
        if self.dataset_type != DatasetType.Text:
            return None

        def collate_batch(batch):
            label_list, text_list = [], []
            for (_label, _text) in batch:
                label_list.append(int(_label))
                processed_text = torch.tensor(self.tokenizer_and_vocab(_text))
                text_list.append(processed_text)
            return text_list, label_list

    # def generate_raw_data(self, phase: MachineLearningPhase):
    #     if self.dataset_type == DatasetType.Vision:
    #         dataset_util = self.get_dataset_util(phase)
    #         return (
    #             (
    #                 dataset_util.get_sample_image(i),
    #                 dataset_util.get_sample_label(i),
    #             )
    #             for i in range(len(dataset_util))
    #         )
    #     raise RuntimeError("Unimplemented Code")

    def get_label_names(self) -> List[str]:
        cache_dir = DatasetCollection.__get_dataset_cache_dir(self.name)
        pickle_file = os.path.join(cache_dir, "label_names.pk")

        def computation_fun():
            if hasattr(self.get_training_dataset(), "classes"):
                return getattr(self.get_training_dataset(), "classes")

            for dataset_type in DatasetType:
                dataset_constructors = DatasetCollection.get_dataset_constructors(
                    dataset_type
                )
                if self.name not in dataset_constructors:
                    continue
                dataset_constructor = dataset_constructors[self.name]
                if hasattr(dataset_constructor, "classes"):
                    return getattr(dataset_constructor, "classes")
                get_logger().error(
                    "%s constructor %s has no classes", self.name, dataset_constructor
                )
                raise NotImplementedError(self.name)
            raise NotImplementedError(self.name)

        return DatasetCollection.__get_cache_data(pickle_file, computation_fun)

    __dataset_root_dir: str = os.path.join(os.path.expanduser("~"), "pytorch_dataset")
    __lock = threading.RLock()

    @staticmethod
    def get_dataset_root_dir():
        with DatasetCollection.__lock:
            return os.getenv(
                "pytorch_dataset_root_dir", DatasetCollection.__dataset_root_dir
            )

    @staticmethod
    def set_dataset_root_dir(root_dir: str):
        with DatasetCollection.__lock:
            DatasetCollection.__dataset_root_dir = root_dir

    @staticmethod
    def __get_dataset_dir(name: str):
        dataset_dir = os.path.join(DatasetCollection.get_dataset_root_dir(), name)
        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir, exist_ok=True)
        return dataset_dir

    @staticmethod
    def __get_dataset_cache_dir(
        name: str, phase: MachineLearningPhase | None = None
    ) -> str:
        cache_dir = os.path.join(DatasetCollection.__get_dataset_dir(name), ".cache")
        if phase is not None:
            cache_dir = os.path.join(cache_dir, str(phase))
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    @staticmethod
    def get_dataset_constructors(dataset_type: DatasetType = None):
        repositories = []
        if dataset_type is None or dataset_type == DatasetType.Vision:
            repositories = [torchvision.datasets, local_vision_datasets]
        elif dataset_type is None or dataset_type == DatasetType.Text:
            repositories = [torchtext.datasets]
        elif dataset_type is None or dataset_type == DatasetType.Audio:
            if has_torchaudio:
                repositories = [torchaudio.datasets, local_audio_datasets]
        dataset_constructors = {}
        for repository in repositories:
            if hasattr(repository, "DATASETS"):
                for name, dataset_constructor in getattr(
                    repository, "DATASETS"
                ).items():
                    if dataset_type == DatasetType.Text:
                        dataset_constructors[name] = dataset_constructor
                        # if hasattr(dataset_constructor, "splits"):
                        #     dataset_constructors[name] = getattr(
                        #         dataset_constructor, "splits"
                        #     )
                        # else:
            for name in dir(repository):
                dataset_constructor = getattr(repository, name)
                if not inspect.isclass(dataset_constructor):
                    continue
                if issubclass(dataset_constructor, torch.utils.data.Dataset):
                    dataset_constructors[name] = dataset_constructor
        return dataset_constructors

    @staticmethod
    def get_by_name(name: str, dataset_kwargs=None):
        with DatasetCollection.__lock:
            all_dataset_constructors = set()
            for dataset_type in DatasetType:
                dataset_constructor = DatasetCollection.get_dataset_constructors(
                    dataset_type
                )
                if name in dataset_constructor:
                    return DatasetCollection.__create_dataset_collection(
                        name, dataset_type, dataset_constructor[name], dataset_kwargs
                    )
                all_dataset_constructors |= dataset_constructor.keys()
            get_logger().error("supported datasets are %s", all_dataset_constructors)
            raise NotImplementedError(name)

    @staticmethod
    def __get_mean_and_std(name: str, dataset):
        cache_dir = os.path.join(DatasetCollection.__get_dataset_dir(name), ".cache")
        pickle_file = os.path.join(cache_dir, "mean_and_std.pk")

        def computation_fun():
            if name.lower() == "imagenet":
                mean = torch.Tensor([0.485, 0.456, 0.406])
                std = torch.Tensor([0.229, 0.224, 0.225])
            else:
                mean, std = DatasetUtil(dataset).get_mean_and_std()
            return (mean, std)

        return DatasetCollection.__get_cache_data(pickle_file, computation_fun)

    @staticmethod
    def __create_dataset_collection(
        name: str, dataset_type: DatasetType, dataset_constructor, dataset_kwargs=None
    ):
        sig = inspect.signature(dataset_constructor)
        dataset_kwargs = DatasetCollection.__prepare_dataset_kwargs(
            name, dataset_type, sig, dataset_kwargs
        )
        training_dataset = None
        validation_dataset = None
        test_dataset = None

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
                            if dataset_type == DatasetType.Text:
                                dataset_kwargs["split"] = "valid"
                            else:
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
                    dataset = dataset_constructor(**dataset_kwargs)
                    if isinstance(
                        dataset, torchtext.data.datasets_utils._RawTextIterableDataset
                    ):
                        dataset = convert_iterable_dataset_to_map(dataset)
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
                    # split = dataset_kwargs.get("split", None)
                    # # if split == "test":
                    # #     break
                    raise e

        if validation_dataset is None or test_dataset is None:
            splited_dataset = None
            if validation_dataset is not None:
                splited_dataset = validation_dataset
                get_logger().warning("split validation dataset for %s", name)
            else:
                splited_dataset = test_dataset
                get_logger().warning("split test dataset for %s", name)
            (
                validation_dataset,
                test_dataset,
            ) = DatasetCollection.__split_for_validation(
                DatasetCollection.__get_dataset_cache_dir(name), splited_dataset
            )
        dc = DatasetCollection(
            training_dataset, validation_dataset, test_dataset, dataset_type, name
        )

        get_logger().info("training_dataset len %s", len(training_dataset))
        get_logger().info("validation_dataset len %s", len(validation_dataset))
        get_logger().info("test_dataset len %s", len(test_dataset))

        if dataset_type == DatasetType.Vision:
            mean, std = DatasetCollection.__get_mean_and_std(
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
        elif dataset_type == DatasetType.Text:

            pass
        # if dataset_type == DatasetType.Audio:
        #     if name == "SPEECHCOMMANDS_SIMPLIFIED":
        #         dc.append_transform(
        #             lambda tensor: torch.nn.ConstantPad1d(
        #                 (0, 16000 - tensor.shape[-1]), 0
        #             )(tensor)
        #         )
        return dc

    @staticmethod
    def __prepare_dataset_kwargs(
        name: str, dataset_type: DatasetType, sig, dataset_kwargs: dict = None
    ):
        if dataset_kwargs is None:
            dataset_kwargs = {}
        if "root" not in dataset_kwargs and "root" in sig.parameters:
            dataset_kwargs["root"] = DatasetCollection.__get_dataset_dir(name)
        if (
            "download" in sig.parameters
            and sig.parameters["download"].default is not None
        ):
            dataset_kwargs["download"] = True

        discarded_dataset_kwargs = set()
        for k in dataset_kwargs:
            if k not in sig.parameters:
                discarded_dataset_kwargs.add(k)
        if discarded_dataset_kwargs:
            get_logger().warning(
                "discarded_dataset_kwargs %s", discarded_dataset_kwargs
            )
            for k in discarded_dataset_kwargs:
                dataset_kwargs.pop(k)
        if dataset_type == DatasetType.Vision:
            if "transform" not in dataset_kwargs:
                dataset_kwargs["transform"] = transforms.Compose(
                    [transforms.ToTensor()]
                )
        return dataset_kwargs

    def __get_label_indices(self, phase):
        with DatasetCollection.__lock:
            cache_dir = DatasetCollection.__get_dataset_cache_dir(self.name, phase)
            pickle_file = os.path.join(cache_dir, "label_indices.pk")
            dataset_util = self.get_dataset_util(phase)
            return DatasetCollection.__get_cache_data(
                pickle_file,
                dataset_util.split_by_label,
            )

    @staticmethod
    def __split_for_validation(cache_dir, splited_dataset):
        pickle_file = os.path.join(cache_dir, "split_index_lists.pk")
        dataset_util = DatasetUtil(splited_dataset)
        split_index_lists = DatasetCollection.__read_data(pickle_file)
        if split_index_lists is not None:
            return dataset_util.split_by_indices(split_index_lists)
        datasets = dataset_util.iid_split([1, 1])
        DatasetCollection.__write_data(pickle_file, [d.indices for d in datasets])
        return datasets

    @staticmethod
    def __get_cache_data(path, computation_fun: Callable):
        with DatasetCollection.__lock:
            data = DatasetCollection.__read_data(path)
            if data is not None:
                return data
            data = computation_fun()
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


class DatasetCollectionConfig:
    def __init__(self, dataset_name=None):
        self.dataset_name = dataset_name
        self.dataset_kwargs = {}
        self.sub_collection_labels = None
        self.training_dataset_percentage = None
        self.training_dataset_indices_path = None
        self.training_dataset_label_map_path = None
        self.training_dataset_label_map = None
        self.training_dataset_label_noise_percentage = None

    def add_args(self, parser):
        parser.add_argument("--dataset_name", type=str, required=True)
        parser.add_argument("--sub_collection_labels", type=str, default=None)
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

    def create_dataset_collection(self, save_dir):
        if self.dataset_name is None:
            raise RuntimeError("dataset_name is None")

        dc = DatasetCollection.get_by_name(self.dataset_name, self.dataset_kwargs)

        # We need to remap labels for the loss function to work
        # if self.sub_collection_labels is not None:
        #     labels = set(self.sub_collection_labels.split("|"))
        #     for phase in MachineLearningPhase:
        #         dc.transform_dataset_to_subset(phase, labels)

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
