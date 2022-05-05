from typing import Callable

from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import TransformType
from torch.utils.data._utils.collate import default_collate


def default_data_extraction(data):
    if len(data) == 3:
        sample_input, target, tmp = data
        return {"input": sample_input, "target": target, "other_info": tmp}
    sample_input, target = data
    return {"input": sample_input, "target": target}


def str_target_to_int(label_names) -> Callable:
    reversed_label_names = {v: k for k, v in label_names.items()}
    get_logger().info("map string targets by %s", reversed_label_names)

    def get_int_target(label_name) -> int:
        return reversed_label_names[label_name]

    return get_int_target


def swap_input_and_target(data):
    data["input"], data["target"] = data["target"], data["input"]
    return data


class Transforms:
    def __init__(self):
        self.__transforms: dict = {}
        self.append(key=TransformType.ExtractData, transform=default_data_extraction)

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

    def transform_inputs(self, inputs: list) -> list:
        inputs = [self.transform_input(i) for i in inputs]
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

    def transform_targets(self, targets: list) -> list:
        targets = [self.transform_target(i) for i in targets]
        return default_collate(targets)
