#!/usr/bin/env python3
from algorithm.quantization.trainer import QuantizationTrainer
from configuration import get_trainer_from_configuration
from dataset import sub_dataset


def test_training():
    trainer = get_trainer_from_configuration("MNIST", "LeNet5")
    trainer.set_training_dataset(sub_dataset(trainer.training_dataset, [0]))
    trainer.hyper_parameter.set_epoch(1)

    trainer = QuantizationTrainer(trainer)
    trainer.train()
