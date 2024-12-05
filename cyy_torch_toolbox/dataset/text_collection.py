import copy
import functools
from typing import Self

from cyy_naive_lib.decorator import Decorator

from ..ml_type import MachineLearningPhase
from .collection import DatasetCollection


class TextDatasetCollection(Decorator):
    __prompt: str | None = None

    def set_prompt(self, prompt: str) -> None:
        assert self.__prompt is None
        self.prompt = prompt
