from typing import Generator

from cyy_naive_lib.log import get_logger

from ..dependency import has_torchvision
from ..ml_type import DatasetType, MachineLearningPhase, TransformType

if has_torchvision:
    import torchvision


class ClassificationDatasetCollection:
    # def __init__(self, dc: DatasetCollection) -> None:
    #     self.__dc = dc

    # def __getattr__(self, name: str) -> Any:
    #     return getattr(self.__dc, name)

    def get_labels(self, use_cache: bool = True) -> set:
        def computation_fun() -> set:
            if self.name is not None and self.name.lower() == "imagenet":
                return set(range(1000))
            return self.get_dataset_util(
                phase=MachineLearningPhase.Training
            ).get_labels()

        if not use_cache:
            return computation_fun()

        return self.get_cached_data("labels.pk", computation_fun)

    def get_label_names(self) -> dict:
        def computation_fun():
            label_names = self.get_dataset_util(
                phase=MachineLearningPhase.Training
            ).get_label_names()
            if not label_names:
                raise NotImplementedError(f"failed to get label names for {self.name}")
            return label_names

        return self.get_cached_data("label_names.pk", computation_fun)

    def get_raw_data(self, phase: MachineLearningPhase, index: int) -> tuple:
        dataset_util = self.get_dataset_util(phase)
        return (
            dataset_util.get_sample_raw_input(index),
            dataset_util.get_sample_label(index),
        )

    def generate_raw_data(self, phase: MachineLearningPhase) -> Generator:
        dataset_util = self.get_dataset_util(phase)
        return (
            self.get_raw_data(phase=phase, index=i) for i in range(len(dataset_util))
        )

    @classmethod
    def get_label(cls, label_name, label_names):
        reversed_label_names = {v: k for k, v in label_names.items()}
        return reversed_label_names[label_name]

    def add_transforms(self, model_evaluator) -> None:
        self.__dc.add_transforms(model_evaluator=model_evaluator)
        # add more transformers for model
        if self.dataset_type == DatasetType.Vision:
            input_size = getattr(
                model_evaluator.get_underlying_model().__class__, "input_size", None
            )
            if input_size is not None:
                get_logger().debug("resize input to %s", input_size)
                self.append_transform(
                    transform=torchvision.transforms.Resize(input_size),
                    key=TransformType.Input,
                )
        get_logger().debug(
            "use transformers for training => \n %s",
            str(self.get_transforms(MachineLearningPhase.Training)),
        )
