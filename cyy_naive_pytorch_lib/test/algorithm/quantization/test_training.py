#!/usr/bin/env python3
from algorithm.quantization.trainer import QuantizationTrainer
from configuration import get_trainer_from_configuration


def test_training():
    trainer = get_trainer_from_configuration("MNIST", "LeNet5")
    trainer.hyper_parameter.set_epoch(1)
    trainer = QuantizationTrainer(trainer)
    trainer.train()
