from typing import Any


from ..ml_type import DatasetType
from .dataset_collection import DatasetCollection


class TextDatasetCollection(DatasetCollection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert self.dataset_type == DatasetType.Text
