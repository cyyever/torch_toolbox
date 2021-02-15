#!/usr/bin/env python3
from algorithm.quantization.trainer import QuantizationTrainer
from basic_trainer import StopTrainingException
from configuration import get_trainer_from_configuration
from model_executor import ModelExecutorCallbackPoint


def stop_training(*args, **kwargs):
    raise StopTrainingException()


def test_training():
    trainer = get_trainer_from_configuration("MNIST", "LeNet5")
    trainer.hyper_parameter.set_epoch(1)
    trainer = QuantizationTrainer(trainer)
    trainer.trainer.add_callback(ModelExecutorCallbackPoint.AFTER_BATCH, stop_training)
    trainer.train()
