import copy
import functools
from typing import Self

from cyy_naive_lib.decorator import Decorator

from ..ml_type import MachineLearningPhase, TransformType
from .collection import DatasetCollection


def str_concat(prefix: str, example: str) -> str:
    return prefix + example


class TextDatasetCollection(Decorator):
    __prompt: str | None = None

    def set_prompt(self, prompt: str) -> None:
        assert self.__prompt is None
        self.__prompt = prompt
        self.set_transform(
            functools.partial(str_concat, prompt),
            key=TransformType.InputTextLast,
        )
