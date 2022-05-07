import copy
import functools
import json
import os
import pickle
import threading
from typing import Callable, Dict

import torch
import torchtext
import torchvision
from cyy_naive_lib.log import get_logger
from ssd_checker import is_ssd
from torch.utils.data._utils.collate import default_collate

from cyy_torch_toolbox.dataset import (convert_iterable_dataset_to_map,
                                       get_dataset_size,
                                       replace_dataset_labels, sub_dataset)
from cyy_torch_toolbox.dataset_repository import get_dataset_constructors
from cyy_torch_toolbox.dataset_transform.transforms import Transforms
from cyy_torch_toolbox.dataset_transform.transforms_factory import \
    add_transforms
from cyy_torch_toolbox.dataset_util import (  # CachedVisionDataset,
    DatasetSplitter, DatasetUtil, TextDatasetUtil, VisionDatasetUtil)
from cyy_torch_toolbox.ml_type import (DatasetType, MachineLearningPhase,
                                       TransformType)
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
        self._datasets: Dict[MachineLearningPhase, torch.utils.data.Dataset] = {}
        self._datasets[MachineLearningPhase.Training] = training_dataset
        self._datasets[MachineLearningPhase.Validation] = validation_dataset
        self._datasets[MachineLearningPhase.Test] = test_dataset
        self.__dataset_type = dataset_type
        self.__transforms: dict[MachineLearningPhase, Transforms] = {}
        for phase in MachineLearningPhase:
            self.__transforms[phase] = Transforms()
        self.__name = name
        self.tokenizer = None

    @property
    def dataset_type(self):
        return self.__dataset_type

    def transform_dataset(
        self, phase: MachineLearningPhase, transformer: Callable
    ) -> None:
        dataset = self.get_dataset(phase)
        dataset_util = self.get_dataset_util(phase)
        self._datasets[phase] = transformer(dataset, dataset_util)

    def foreach_dataset(self):
        for phase in MachineLearningPhase:
            yield self.get_dataset(phase=phase)

    def transform_all_datasets(self, transformer: Callable) -> None:
        for phase in MachineLearningPhase:
            self.transform_dataset(phase, transformer)

    def get_dataset(self, phase: MachineLearningPhase) -> torch.utils.data.Dataset:
        return self._datasets[phase]

    def get_training_dataset(self) -> torch.utils.data.Dataset:
        return self.get_dataset(MachineLearningPhase.Training)

    def get_transforms(self, phase) -> Transforms:
        return self.__transforms[phase]

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
            dataset=self.get_dataset(phase),
            transforms=self.__transforms[phase],
            name=self.name,
        )

    def clear_transform(self, key, phases=None):
        for phase in MachineLearningPhase:
            if phases is not None and phase not in phases:
                continue
            self.__transforms[phase].clear(key)

    def append_transform(self, transform, key=TransformType.Input, phases=None):
        for phase in MachineLearningPhase:
            if phases is not None and phase not in phases:
                continue
            self.__transforms[phase].append(key, transform)

    @property
    def name(self) -> str:
        return self.__name

    def transform_text(self, phase, text):
        return self.get_transforms(phase).transform_text(text)

    def collate_batch(self, batch, phase):
        inputs = []
        targets = []
        other_info = []
        for item in batch:
            res = self.__transforms[phase].extract_data(item)
            inputs.append(res["input"])
            targets.append(res["target"])
            if "other_info" in res:
                other_info.append(res["other_info"])
        inputs = self.__transforms[phase].transform_inputs(inputs)
        targets = self.__transforms[phase].transform_targets(targets)

        # TODO for classification
        targets = targets.reshape(-1)
        batch_size = len(batch)
        if other_info:
            other_info = default_collate(other_info)
            return {"size": batch_size, "content": (inputs, targets, other_info)}
        return {"size": batch_size, "content": (inputs, targets)}

    __dataset_root_dir: str = os.path.join(os.path.expanduser("~"), "pytorch_dataset")
    lock = threading.RLock()

    @classmethod
    def get_dataset_root_dir(cls):
        with cls.lock:
            return os.getenv("pytorch_dataset_root_dir", cls.__dataset_root_dir)

    @classmethod
    def set_dataset_root_dir(cls, root_dir: str):
        with cls.lock:
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
    def create(
        cls,
        name: str,
        dataset_type: DatasetType,
        dataset_constructor,
        dataset_kwargs: dict = None,
    ) -> tuple:
        constructor_kwargs = get_kwarg_names(dataset_constructor)
        dataset_kwargs_fun = cls.__prepare_dataset_kwargs(
            name, constructor_kwargs, dataset_kwargs
        )
        training_dataset = None
        validation_dataset = None
        test_dataset = None

        for phase in MachineLearningPhase:
            while True:
                try:
                    processed_dataset_kwargs = dataset_kwargs_fun(
                        phase=phase, dataset_type=dataset_type
                    )
                    if processed_dataset_kwargs is None:
                        break
                    dataset = dataset_constructor(**processed_dataset_kwargs)
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
                    if "for argument split. Valid values are" in str(e):
                        break
                    if "Unknown split" in str(e):
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
        assert not (validation_dataset is None and test_dataset is None)

        if validation_dataset is None or test_dataset is None:
            if validation_dataset is not None:
                splitted_dataset = validation_dataset
                get_logger().debug("split validation dataset for %s", name)
            else:
                splitted_dataset = test_dataset
                get_logger().debug("split test dataset for %s", name)
            (validation_dataset, test_dataset,) = cls.__split_for_validation(
                cls.__get_dataset_cache_dir(name), splitted_dataset
            )

        try:
            get_logger().info(
                "training_dataset len %s", get_dataset_size(training_dataset)
            )
            get_logger().info(
                "validation_dataset len %s", get_dataset_size(validation_dataset)
            )
            get_logger().info("test_dataset len %s", get_dataset_size(test_dataset))
        except BaseException:
            pass
        return (training_dataset, validation_dataset, test_dataset, dataset_type, name)

    def is_classification_dataset(self) -> bool:
        first_target = self.get_dataset_util(
            phase=MachineLearningPhase.Training
        ).get_sample_label(0)
        match first_target:
            case int():
                return True
            case "pos" | "neg":
                return True
        return False

    @classmethod
    def __prepare_dataset_kwargs(
        cls,
        name: str,
        constructor_kwargs: set,
        dataset_kwargs: dict = None,
    ) -> Callable:
        if dataset_kwargs is None:
            dataset_kwargs = {}
            new_dataset_kwargs = {}
        else:
            new_dataset_kwargs = copy.deepcopy(dataset_kwargs)
        if "root" not in new_dataset_kwargs:
            new_dataset_kwargs["root"] = cls.__get_dataset_dir(name)
        if "download" not in new_dataset_kwargs:
            new_dataset_kwargs["download"] = True

        def get_dataset_kwargs_per_phase(
            dataset_type: DatasetType, phase: MachineLearningPhase
        ) -> dict | None:
            if "train" in constructor_kwargs:
                # Some dataset only have train and test parts
                if phase == MachineLearningPhase.Validation:
                    return None
                new_dataset_kwargs["train"] = phase == MachineLearningPhase.Training
            elif "split" in constructor_kwargs:
                if phase == MachineLearningPhase.Training:
                    new_dataset_kwargs["split"] = dataset_kwargs.get(
                        "train_split", "train"
                    )
                elif phase == MachineLearningPhase.Validation:
                    if "val_split" in dataset_kwargs:
                        new_dataset_kwargs["split"] = dataset_kwargs["val_split"]
                    else:
                        if dataset_type == DatasetType.Text:
                            new_dataset_kwargs["split"] = "valid"
                        else:
                            new_dataset_kwargs["split"] = "val"
                else:
                    new_dataset_kwargs["split"] = dataset_kwargs.get(
                        "test_split", "test"
                    )
            elif "subset" in constructor_kwargs:
                if phase == MachineLearningPhase.Training:
                    new_dataset_kwargs["subset"] = "training"
                elif phase == MachineLearningPhase.Validation:
                    new_dataset_kwargs["subset"] = "validation"
                else:
                    new_dataset_kwargs["subset"] = "testing"
            discarded_dataset_kwargs = set()
            for k in new_dataset_kwargs:
                if k not in constructor_kwargs:
                    discarded_dataset_kwargs.add(k)
            if discarded_dataset_kwargs:
                get_logger().warning(
                    "discarded_dataset_kwargs %s", discarded_dataset_kwargs
                )
                for k in discarded_dataset_kwargs:
                    new_dataset_kwargs.pop(k)
            return new_dataset_kwargs

        return get_dataset_kwargs_per_phase

    @staticmethod
    def __split_for_validation(cache_dir, splitted_dataset):
        pickle_file = os.path.join(cache_dir, "split_index_lists.pk")
        dataset_util = DatasetSplitter(
            dataset=splitted_dataset, transforms=Transforms()
        )
        split_index_lists = DatasetCollection.__read_data(pickle_file)
        if split_index_lists is not None:
            return dataset_util.split_by_indices(split_index_lists)
        datasets = dataset_util.iid_split([1, 1])
        DatasetCollection.__write_data(pickle_file, [d.indices for d in datasets])
        return datasets

    def _get_cache_data(self, path: str, computation_fun: Callable) -> dict:
        with DatasetCollection.lock:
            cache_dir = DatasetCollection.__get_dataset_cache_dir(self.name)
            path = os.path.join(cache_dir, path)
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


class ClassificationDatasetCollection(DatasetCollection):
    @classmethod
    def create(cls, **kwargs):
        dataset_kwargs = kwargs.get("dataset_kwargs", {})
        if not dataset_kwargs:
            dataset_kwargs = {}
        dc: ClassificationDatasetCollection = cls(*DatasetCollection.create(**kwargs))
        add_transforms(dc, dataset_kwargs)
        return dc

    def get_mean_and_std(self, dataset):
        transforms = Transforms()
        transforms.append(
            key=TransformType.Input, transform=torchvision.transforms.ToTensor()
        )

        def computation_fun():
            return VisionDatasetUtil(
                dataset=dataset,
                transforms=transforms,
                name=self.name,
            ).get_mean_and_std()

        return self._get_cache_data("mean_and_std.pk", computation_fun)

    def get_labels(self, use_cache: bool = True) -> set:
        def computation_fun():
            if self.name.lower() == "imagenet":
                return range(1000)
            return self.get_dataset_util(
                phase=MachineLearningPhase.Training
            ).get_labels()

        if not use_cache:
            return computation_fun()

        return self._get_cache_data("labels.pk", computation_fun)

    def get_label_names(self) -> dict:
        def computation_fun():
            label_names = self.get_dataset_util(
                phase=MachineLearningPhase.Training
            ).get_label_names()
            if not label_names:
                raise NotImplementedError(f"failed to get label names for {self.name}")
            return label_names

        return self._get_cache_data("label_names.pk", computation_fun)

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
                for i in range(len(dataset_util))
            )
        raise RuntimeError("Unimplemented Code")

    @classmethod
    def get_label(cls, label_name, label_names):
        reversed_label_names = {v: k for k, v in label_names.items()}
        return reversed_label_names[label_name]

    def adapt_to_model(self, model, model_kwargs=None):
        """add more transformers for model"""
        if self.dataset_type == DatasetType.Vision:
            input_size = getattr(model.__class__, "input_size", None)
            if input_size is not None:
                get_logger().debug("resize input to %s", input_size)
                self.append_transform(torchvision.transforms.Resize(input_size))
            return
        if self.dataset_type == DatasetType.Text:
            max_len = model_kwargs.get("max_len", None)
            if max_len is not None:
                get_logger().debug("resize input to %s", max_len)
                self.append_transform(
                    torchtext.transforms.Truncate(max_seq_len=max_len),
                    key=TransformType.Input,
                )
        get_logger().debug(
            "use transformers for training => \n %s",
            str(self.get_transforms(MachineLearningPhase.Training)),
        )


def create_dataset_collection(cls, name: str, dataset_kwargs: dict = None):
    with cls.lock:
        all_dataset_constructors = set()
        for dataset_type in DatasetType:
            dataset_constructor = get_dataset_constructors(dataset_type)
            if name in dataset_constructor:
                return cls.create(
                    name=name,
                    dataset_type=dataset_type,
                    dataset_constructor=dataset_constructor[name],
                    dataset_kwargs=dataset_kwargs,
                )
            all_dataset_constructors |= dataset_constructor.keys()
        get_logger().error("supported datasets are %s", all_dataset_constructors)
        raise NotImplementedError(name)


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
        parser.add_argument("--dataset_kwarg_json_path", type=str, default=None)

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
        if args.dataset_kwarg_json_path is not None:
            with open(args.dataset_kwarg_json_path, "rt", encoding="utf-8") as f:
                self.dataset_kwargs |= json.load(f)

    def create_dataset_collection(self, save_dir=None):
        if self.dataset_name is None:
            raise RuntimeError("dataset_name is None")

        dc = create_dataset_collection(
            ClassificationDatasetCollection, self.dataset_name, self.dataset_kwargs
        )
        if not dc.is_classification_dataset():
            dc = create_dataset_collection(
                DatasetCollection, self.dataset_name, self.dataset_kwargs
            )

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
