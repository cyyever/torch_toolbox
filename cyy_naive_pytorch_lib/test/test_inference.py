#!/usr/bin/env python3
from configuration import get_inferencer_from_configuration
from ml_type import MachineLearningPhase


def test_training():
    inferencer = get_inferencer_from_configuration(
        "MNIST", "LeNet5", MachineLearningPhase.Test
    )
    inferencer.inference()
