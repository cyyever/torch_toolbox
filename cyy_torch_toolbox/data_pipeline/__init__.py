from typing import Any

from ..ml_type import Factory
from .dataset import *
from .transform import *

global_data_transform_factory = Factory()


def append_transforms_to_dc(dc: Any, model_evaluator: Any = None) -> None:
    constructor = global_data_transform_factory.get(dc.dataset_type)
    assert constructor is not None
    return constructor(dc=dc, model_evaluator=model_evaluator)
