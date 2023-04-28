from ..ml_type import DatasetType


def add_data_extraction(dc):
    if dc.dataset_type == DatasetType.Text:
        from .text import add_text_extraction

        add_text_extraction(dc=dc)
        return
    if dc.dataset_type == DatasetType.Vision:
        from .vision import add_vision_extraction

        add_vision_extraction(dc=dc)
        return


def add_transforms(dc, dataset_kwargs, model_evaluator):
    if dc.dataset_type == DatasetType.Vision:
        from .vision import add_vision_transforms

        add_vision_transforms(dc=dc)
        return
    if dc.dataset_type == DatasetType.Text:
        from .text import add_text_transforms

        add_text_transforms(
            dc=dc, dataset_kwargs=dataset_kwargs, model_evaluator=model_evaluator
        )
        return
