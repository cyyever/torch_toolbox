from cyy_torch_toolbox import hyper_parameter


def test_hyper_parameter() -> None:
    names = hyper_parameter.HyperParameter.get_optimizer_names()
    assert names
    print(names)
    names = hyper_parameter.HyperParameter.get_lr_scheduler_names()
    assert names
    print(names)
