#!/usr/bin/env python3
from algorithm.quantization.qat import QuantizationAwareTraining
from configuration import get_trainer_from_configuration
from ml_type import StopExecutingException
from model_executor import ModelExecutorCallbackPoint


def stop_training(*args, **kwargs):
    raise StopExecutingException()


def test_training():
    trainer = get_trainer_from_configuration("MNIST", "LeNet5")
    trainer.hyper_parameter.set_epoch(1)
    qat = QuantizationAwareTraining()
    qat.append_to_model_executor(trainer)
    trainer.add_callback(ModelExecutorCallbackPoint.AFTER_BATCH, stop_training)
    trainer.train()
