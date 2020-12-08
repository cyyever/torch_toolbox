#!/usr/bin/env python3

import dataset
from configuration import  get_trainer_from_configuration


def test_training():
    trainer=get_trainer_from_configuration("MNIST","LeNet5")
    trainer.hyper_parameter.set_epochs(1)
    trainer.train()
