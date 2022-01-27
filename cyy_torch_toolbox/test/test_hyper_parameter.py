#!/usr/bin/env python3

import hyper_parameter


def test_hyper_parameter():
    res = hyper_parameter.get_recommended_hyper_parameter("MNIST", "")
    assert res is not None
    names = hyper_parameter.HyperParameter.get_optimizer_names()
    assert names
    print(names)
    names = hyper_parameter.HyperParameter.get_lr_scheduler_names()
    assert names
    print(names)
