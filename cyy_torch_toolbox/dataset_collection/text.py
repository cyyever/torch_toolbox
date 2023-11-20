from typing import Any

from cyy_naive_lib.log import get_logger

from ..ml_type import DatasetType
from ..tokenizer import get_tokenizer
from .dataset_collection import DatasetCollection


class TextDatasetCollection(DatasetCollection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert self.dataset_type == DatasetType.Text
        self.__tokenizer: Any | None = None
        self.__model_kwargs: dict = {}

    def set_model_kwargs(self, model_kwargs: dict) -> None:
        self.__model_kwargs = model_kwargs

    @property
    def tokenizer(self) -> Any | None:
        if self.__tokenizer is None:
            tokenizer_kwargs = self.dataset_kwargs.get("tokenizer", {})
            if (
                "type" not in tokenizer_kwargs
                and "hugging_face" in self.__model_kwargs.get("name", "")
            ):
                tokenizer_kwargs["type"] = "hugging_face"
            assert "type" in tokenizer_kwargs
            self.__tokenizer = get_tokenizer(self, tokenizer_kwargs)
            get_logger().info("tokenizer is %s", self.__tokenizer)
            assert self.__tokenizer is not None
        return self.__tokenizer

    # def __copy__(self):
    #     new_obj = super().__copy__()
    #     new_obj.__model_kwargs = copy.copy(self.__model_kwargs)
    #     new_obj.__tokenizer = copy.copy(self.__tokenizer)
    #     return new_obj
