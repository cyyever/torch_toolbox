from cyy_torch_toolbox.ml_type import DatasetType


def add_data_extraction(dc) -> None:
    if dc.dataset_type == DatasetType.Text:
        from .text import add_text_extraction

        add_text_extraction(dc=dc)
        return
    if dc.dataset_type == DatasetType.Vision:
        from .vision import add_vision_extraction

        add_vision_extraction(dc=dc)
        return


def add_transforms(dc, model_evaluator) -> None:
    if dc.dataset_type == DatasetType.Vision:
        from .vision import add_vision_transforms

        add_vision_transforms(dc=dc, model_evaluator=model_evaluator)
        return
    if dc.dataset_type == DatasetType.Text:
        from .text import add_text_transforms

        add_text_transforms(dc=dc, model_evaluator=model_evaluator)
        return
