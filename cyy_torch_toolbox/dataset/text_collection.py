import functools
from cyy_naive_lib.log import log_debug

from ..data_pipeline import DataPipeline, Transform
from .collection import DatasetCollection


def format_prompt(prompt: str, example: str | dict) -> str | dict:
    if isinstance(example, str):
        log_debug("final input is %s", prompt + example)
        return prompt + example
    example["input"] = prompt.format(**example)
    log_debug("final input is %s", example["input"])
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
                Transform(fun=functools.partial(format_prompt, self.prompt))
            )
        return self.__text_pipeline
