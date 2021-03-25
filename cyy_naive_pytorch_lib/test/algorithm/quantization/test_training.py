#!/usr/bin/env python3
from algorithm.quantization.qat import QuantizationAwareTraining
from default_config import DefaultConfig


def test_training():
    trainer = DefaultConfig("MNIST", "LeNet5").create_trainer()
    trainer.hyper_parameter.set_epoch(1)
    trainer.hyper_parameter.set_learning_rate(0.01)
    qat = QuantizationAwareTraining()
    qat.append_to_model_executor(trainer)
    trainer.train()
