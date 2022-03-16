#!/usr/bin/env python3
from default_config import DefaultConfig


def test_vision_training():
    config = DefaultConfig(dataset_name="MNIST", model_name="LeNet5")
    config.hyper_parameter_config.epoch = 1
    trainer = config.create_trainer()
    trainer.train()


def test_text_training():
    config = DefaultConfig(dataset_name="IMDB", model_name="simplelstm")
    config.hyper_parameter_config.epoch = 1
    trainer = config.create_trainer()
    trainer.train()
