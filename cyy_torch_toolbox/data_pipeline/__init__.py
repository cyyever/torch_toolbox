from cyy_torch_toolbox.ml_type import DatasetType

from ..factory import Factory

global_data_transform_factory = Factory()


def append_transforms_to_dc(dc, model_evaluator=None) -> None:
    constructor = global_data_transform_factory.get(dc.dataset_type)
    if constructor is not None:
        return constructor(dc=dc, model_evaluator=model_evaluator)

    if dc.dataset_type == DatasetType.Vision:
        from .vision import add_vision_extraction, add_vision_transforms

        if model_evaluator is None:
            add_vision_extraction(dc=dc)
        else:
            add_vision_transforms(dc=dc, model_evaluator=model_evaluator)
        return
