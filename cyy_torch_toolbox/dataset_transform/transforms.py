from typing import Callable

from cyy_torch_toolbox.ml_type import TransformType
from torch.utils.data._utils.collate import default_collate


def default_data_extraction(data):
    if len(data) == 3:
        input, target, tmp = data
        return {"input": input, "target": target, "other_info": tmp}
    input, target = data
    return {"input": input, "target": target}


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

    def transform_inputs(self, inputs: list) -> list:
        for f in self.get(TransformType.InputText):
            inputs = [f(i) for i in inputs]
        for f in self.get(TransformType.Input):
            inputs = [f(i) for i in inputs]
        batch_transforms = self.get(TransformType.InputBatch)
        if not batch_transforms:
            batch_transforms.append(default_collate)
        for f in batch_transforms:
            inputs = f(inputs)
        return inputs

    def transform_targets(self, targets: list) -> list:
        for f in self.get(TransformType.Target):
            targets = [f(i) for i in targets]
        return default_collate(targets)
