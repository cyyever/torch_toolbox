#!/usr/bin/env python3
from default_config import DefaultConfig
from ml_type import MachineLearningPhase


def test_training():
    config = DefaultConfig(dataset_name="MNIST", model_name="LeNet5")
    config.hyper_parameter_config.epoch = 1
    trainer = config.create_trainer()
    inferencer = trainer.get_inferencer(MachineLearningPhase.Test)
    inferencer.inference()
