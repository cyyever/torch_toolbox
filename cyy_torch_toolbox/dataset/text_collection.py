import functools

from cyy_naive_lib.decorator import Decorator

from ..ml_type import TransformType


def str_concat(prefix: str, example: str) -> str:
    return prefix + example


class TextDatasetCollection(Decorator):
    __prompt: str | None = None

    @property
    def prompt(self) -> str | None:
        return self.__prompt

    def set_prompt(self, prompt: str) -> None:
        assert self.__prompt is None
        self.__prompt = prompt
        self.set_transform(
            functools.partial(str_concat, prompt),
            key=TransformType.InputTextLast,
        )
