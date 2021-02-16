#!/usr/bin/env python3
from configuration import get_trainer_from_configuration
from hyper_parameter import HyperParameterAction
from ml_types import StopExecutingException
from model_executor import ModelExecutorCallbackPoint


def stop_training(*args, **kwargs):
    raise StopExecutingException()


def test_training():
    trainer = get_trainer_from_configuration("MNIST", "LeNet5")
    trainer.hyper_parameter.set_learning_rate(HyperParameterAction.FIND_LR)
    trainer.hyper_parameter.set_epoch(1)
    trainer.add_callback(ModelExecutorCallbackPoint.AFTER_BATCH, stop_training)
    trainer.train()
