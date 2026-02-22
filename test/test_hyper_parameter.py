from cyy_torch_toolbox import get_optimizer_names, hyper_parameter


def test_optimizer_names() -> None:
    names = get_optimizer_names()
    assert "AdamW" in names
    assert "SGD" in names


def test_lr_scheduler_names() -> None:
    names = hyper_parameter.HyperParameter.get_lr_scheduler_names()
    assert "Optimizer" not in names
    assert "ReduceLROnPlateau" in names
    assert "CosineAnnealingLR" in names
