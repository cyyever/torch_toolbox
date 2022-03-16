#!/usr/bin/env python3
from cyy_torch_toolbox.device import CudaDeviceGreedyAllocator
from default_config import DefaultConfig


def test_vision_training():
    config = DefaultConfig(dataset_name="MNIST", model_name="LeNet5")
    config.hyper_parameter_config.epoch = 1
    trainer = config.create_trainer()
    trainer.train()


def test_text_training():
    device = CudaDeviceGreedyAllocator().get_device(
        max_needed_bytes=9 * 1024 * 1024 * 1024
    )
    if device is None:
        return
    config = DefaultConfig(dataset_name="IMDB", model_name="simplelstm")
    config.hyper_parameter_config.epoch = 1
    trainer = config.create_trainer()
    trainer.train()
