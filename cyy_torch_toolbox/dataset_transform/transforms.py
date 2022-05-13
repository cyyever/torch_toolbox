import copy
import functools
from typing import Any, Callable

from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.dataset import get_dataset_size
from cyy_torch_toolbox.ml_type import TransformType
from torch.utils.data._utils.collate import default_collate


def default_data_extraction(data: Any) -> dict:
    match data:
        case {"data": real_data, "index": index}:
            return default_data_extraction(real_data) | {"index": index}
        case[sample_input, target]:
            return {"input": sample_input, "target": target}
    raise NotImplementedError()


def __get_int_target(reversed_label_names, label_name: str) -> int:
    return reversed_label_names[label_name]


def str_target_to_int(label_names) -> Callable:
    reversed_label_names = {v: k for k, v in label_names.items()}
    get_logger().info("map string targets by %s", reversed_label_names)
    return functools.partial(__get_int_target, reversed_label_names)


def swap_input_and_target(data):
    data["input"], data["target"] = data["target"], data["input"]
    return data


class Transforms:
    def __init__(self):
        self.__transforms: dict = {}
        self.append(key=TransformType.ExtractData, transform=default_data_extraction)

    def clear(self, key: TransformType) -> None:
        if key in self.__transforms:
            self.__transforms[key] = []

    def insert(self, key: TransformType, idx: int, transform: Callable) -> None:
        if key not in self.__transforms:
            self.__transforms[key] = []
        self.__transforms[key].insert(idx, transform)

    def append(self, key: TransformType, transform: Callable) -> None:
        if key not in self.__transforms:
            self.__transforms[key] = []
        self.__transforms[key].append(transform)

    def get(self, key: TransformType) -> list:
        return self.__transforms.get(key, [])

    def transform_text(self, text):
        for f in self.get(TransformType.InputText):
            text = f(text)
        return text

    def extract_data(self, data):
        for f in self.get(TransformType.ExtractData):
            data = f(data)
        return data

    def transform_input(self, sample_input):
        sample_input = self.transform_text(sample_input)
        for f in self.get(TransformType.Input):
            sample_input = f(sample_input)
        return sample_input

    def random_transform_input(self, sample_input):
        for f in self.get(TransformType.RandomInput):
            sample_input = f(sample_input)
        return sample_input

    def transform_inputs(self, inputs: list) -> list:
        batch_transforms = self.get(TransformType.InputBatch)
        if not batch_transforms:
            batch_transforms.append(default_collate)
        for f in batch_transforms:
            inputs = f(inputs)
        return inputs

    def transform_target(self, target):
        for f in self.get(TransformType.Target):
            target = f(target)
        return target

    def transform_targets(self, targets: list):
        batch_transforms = self.get(TransformType.TargetBatch)
        if not batch_transforms:
            batch_transforms.append(default_collate)
        for f in batch_transforms:
            targets = f(targets)
        return targets.reshape(-1)

    def collate_batch(self, batch):
        inputs = []
        targets = []
        other_info = []
        for data in batch:
            data = copy.copy(self.extract_data(data))
            sample_input = self.transform_input(data.pop("input"))
            sample_input = self.random_transform_input(sample_input)
            inputs.append(sample_input)
            targets.append(self.transform_target(data.pop("target")))
            other_info.append(data)
        inputs = self.transform_inputs(inputs)
        targets = self.transform_targets(targets)
        batch_size = len(batch)
        if other_info:
            other_info = default_collate(other_info)
            return {"size": batch_size, "data": (inputs, targets, other_info)}
        return {"size": batch_size, "data": (inputs, targets)}

    def cache_transforms(self, dataset) -> tuple[dict, Any]:
        transformed_dataset = {}
        for k in range(get_dataset_size(dataset)):
            item = self.extract_data(dataset[k])
            item["input"] = self.transform_input(item["input"])
            item["target"] = self.transform_target(item["target"])
            transformed_dataset[k] = item
        new_transforms = copy.deepcopy(self)
        new_transforms.clear(TransformType.ExtractData)
        new_transforms.clear(TransformType.InputText)
        new_transforms.clear(TransformType.Input)
        new_transforms.clear(TransformType.Target)
        return transformed_dataset, new_transforms

    def __str__(self):
        desc = []
        for k in (
            TransformType.ExtractData,
            TransformType.InputText,
            TransformType.Input,
            TransformType.RandomInput,
            TransformType.InputBatch,
            TransformType.Target,
        ):
            if k in self.__transforms and self.__transforms[k]:
                desc.append(str(k) + "=>")
                for t in self.__transforms[k]:
                    desc.append(str(t))
        return "\n".join(desc)
