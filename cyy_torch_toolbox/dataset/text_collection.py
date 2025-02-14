import functools
from cyy_naive_lib.log import log_debug, log_error

from ..data_pipeline import DataPipeline, Transform
from .collection import DatasetCollection


def format_prompt(prompt: str, example: str | dict) -> str | dict:
    if isinstance(example, str):
        log_debug("final input is %s", prompt + example)
        return prompt + example
    try:
        extra_kwargs = {}
        for k, v in example.items():
            new_k = f"space_join_{k}"
            if new_k in prompt:
                extra_kwargs[new_k] = " ".join([str(a) for a in v])
            new_k = f"comma_join_{k}"
            if new_k in prompt:
                extra_kwargs[new_k] = ",".join([str(a) for a in v])
        example["input"] = prompt.format(**example, **extra_kwargs)
    except BaseException as e:
        log_error("formatting fail")
        log_error("prompt is:\n%s", prompt)
        log_error("input keys are:\n%s", example.keys())
        raise e
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
