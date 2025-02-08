import functools

from ..data_pipeline import (
    DataPipeline,
    Transform,
)
from .collection import DatasetCollection


def str_concat(prefix: str, example: str | dict) -> str | dict:
    if isinstance(example, str):
        return prefix + example
    example["input"] = prefix + example["input"]
    return example


class TextDatasetCollection(DatasetCollection):
    __prompt: str | None = None
    __text_pipeline: DataPipeline | None = None

    @property
    def prompt(self) -> str | None:
        return self.__prompt

    def set_prompt(self, prompt: str) -> None:
        assert self.__prompt is None
        self.__prompt = prompt

    def append_text_transform(self, transform: Transform) -> None:
        if self.__text_pipeline is None:
            self.__text_pipeline = DataPipeline()
        self.__text_pipeline.append(transform)

    def get_text_pipeline(self) -> DataPipeline | None:
        if self.prompt is not None:
            self.append_text_transform(
                Transform(fun=functools.partial(str_concat, self.prompt))
            )
        return self.__text_pipeline
