#!/usr/bin/env python3
from config import Config


def test_training():
    config = Config(dataset_name="MNIST", model_name="LeNet5")
    config.epoch = 1
    trainer = config.create_trainer()
    trainer.train()
