import copy
from typing import Any

from ..ml_type import DatasetType
from ..tokenizer import get_tokenizer
from .dataset_collection import DatasetCollection


class TextDatasetCollection(DatasetCollection):
    def __init__(self, dc: DatasetCollection) -> None:
        assert dc.dataset_type == DatasetType.Text
        self.__dc = dc
        self.__tokenizer: Any | None = None

    def __getattr__(self, name: str):
        if name == "__copy__" or (not name.startswith("_") and hasattr(self, name)):
            return getattr(self, name)
        return getattr(self.__dc, name)

    @property
    def tokenizer(self) -> Any | None:
        if self.__tokenizer is None:
            self.__tokenizer = get_tokenizer(
                self, self.__dataset_kwargs.get("tokenizer", {})
            )
        return self.__tokenizer

    def __copy__(self):
        new_obj = super().__copy__(self)
        new_obj.__tokenizer = copy.copy(self.__tokenizer)
        return new_obj
