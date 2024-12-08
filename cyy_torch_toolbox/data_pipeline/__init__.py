from typing import Any

from ..ml_type import Factory
from .dataset import get_dataset_size, select_item, subset_dp, DatasetWithIndex
from .pipeline import DataPipeline
from .transform import Transform, BatchTransform, DatasetTransform

global_data_transform_factory = Factory()


def append_transforms_to_dc(dc: Any, model_evaluator: Any = None) -> None:
    constructors = global_data_transform_factory.get(dc.dataset_type)
    if not isinstance(constructors, list):
        assert constructors is not None
        constructors = [constructors]
    for constructor in constructors:
        constructor(dc=dc, model_evaluator=model_evaluator)


__all__ = [
    "Transform",
    "BatchTransform",
    "DatasetTransform",
    "DataPipeline",
    "Factory",
    "get_dataset_size",
    "select_item",
    "subset_dp",
    "DatasetWithIndex",
]
