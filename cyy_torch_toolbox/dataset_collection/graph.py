from typing import Any

from ..ml_type import DatasetType, MachineLearningPhase
from .dataset_collection import DatasetCollection


class GraphDatasetCollection(DatasetCollection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert self.dataset_type == DatasetType.Graph

    def get_edge_dict(self) -> set:
        def computation_fun() -> set:
            return self.get_dataset_util(
                phase=MachineLearningPhase.Training
            ).get_edge_dict()

        if not use_cache:
            return computation_fun()

        return self.get_cached_data("edge_idx.pk", computation_fun)
