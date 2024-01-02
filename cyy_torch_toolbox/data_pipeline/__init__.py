from ..factory import Factory
from .transform import *

global_data_transform_factory = Factory()


def append_transforms_to_dc(dc, model_evaluator=None) -> None:
    constructor = global_data_transform_factory.get(dc.dataset_type)
    assert constructor is not None
    return constructor(dc=dc, model_evaluator=model_evaluator)
