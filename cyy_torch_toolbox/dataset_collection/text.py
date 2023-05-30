import copy
from typing import Any

from ..ml_type import DatasetType
from ..tokenizer import get_tokenizer
from .dataset_collection import DatasetCollection


class TextDatasetCollection(DatasetCollection):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self.dataset_type == DatasetType.Text
        self.__tokenizer: Any | None = None
        self.__model_kwargs: dict | None = None

    def set_model_kwargs(self, model_kwargs: dict) -> None:
        self.__model_kwargs = model_kwargs

    @property
    def tokenizer(self) -> Any | None:
        if self.__tokenizer is None:
            tokenizer_kwargs = self.__dataset_kwargs.get("tokenizer", {})
            self.__tokenizer = get_tokenizer(self, tokenizer_kwargs)
        return self.__tokenizer

    def __copy__(self):
        new_obj = super().__copy__()
        new_obj.__tokenizer = copy.copy(self.__tokenizer)
        return new_obj
