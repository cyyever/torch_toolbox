#!/usr/bin/env python3
from cyy_torch_toolbox.device import CudaDeviceGreedyAllocator
from cyy_torch_toolbox.ml_type import (ModelExecutorHookPoint,
                                       StopExecutingException)
from default_config import DefaultConfig


def stop_training(*args, **kwargs):
    raise StopExecutingException()


def test_vision_training():
    config = DefaultConfig(dataset_name="MNIST", model_name="LeNet5")
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.learning_rate = 0.01
    trainer = config.create_trainer()
    trainer.insert_callback(
        -1, ModelExecutorHookPoint.AFTER_BATCH, "stop_training", stop_training
    )
    trainer.train()


def test_text_training():
    device = CudaDeviceGreedyAllocator().get_device(
        max_needed_bytes=9 * 1024 * 1024 * 1024
    )
    if device is None:
        return
    config = DefaultConfig(dataset_name="IMDB", model_name="simplelstm")
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.learning_rate = 0.01
    trainer = config.create_trainer()
    trainer.insert_callback(
        -1, ModelExecutorHookPoint.AFTER_BATCH, "stop_training", stop_training
    )
    trainer.train()
