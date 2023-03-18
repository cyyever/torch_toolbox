#!/usr/bin/env python3
from cyy_torch_toolbox.default_config import DefaultConfig
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.dependency import has_torchvision,has_torchtext


def test_inference():
    if not has_torchvision:
        return
    config = DefaultConfig(dataset_name="MNIST", model_name="LeNet5")
    config.hyper_parameter_config.epoch = 1
    trainer = config.create_trainer()
    inferencer = trainer.get_inferencer(MachineLearningPhase.Test)
    inferencer.inference()


def test_gradient():
    if not has_torchtext:
        return
    config = DefaultConfig(dataset_name="IMDB", model_name="simplelstm")
    config.hyper_parameter_config.epoch = 1
    trainer = config.create_trainer()
    inferencer = trainer.get_inferencer(MachineLearningPhase.Test)
    inferencer.get_gradient()
