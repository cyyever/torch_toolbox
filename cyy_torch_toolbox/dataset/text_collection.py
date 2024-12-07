import functools

from cyy_naive_lib.decorator import Decorator

from ..data_pipeline import (
    DataPipeline,
    Transform,
    append_transforms_to_dc,
)


def str_concat(prefix: str, example: str) -> str:
    return prefix + example


class TextDatasetCollection(Decorator):
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

    def get_text_pipeline(self) -> DataPipeline:
        if self.prompt is not None:
            self.append_text_transform(
                Transform(fun=functools.partial(str_concat, self.prompt))
            )
        assert self.__text_pipeline is not None
        return self.__text_pipeline

    def add_data_pipeline(self, model_evaluator) -> None:
        append_transforms_to_dc(dc=self, model_evaluator=model_evaluator)
