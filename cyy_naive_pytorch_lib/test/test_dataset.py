#!/usr/bin/env python3

import dataset
from dataset_collection import DatasetCollection
from ml_type import MachineLearningPhase


def test_dataset():
    mnist = DatasetCollection.get_by_name("MNIST")
    mnist_training = mnist.get_dataset(MachineLearningPhase.Training)
    assert (
        abs(
            len(mnist_training)
            / len(mnist.get_dataset(MachineLearningPhase.Validation))
            - 12
        )
        < 0.01
    )
    assert dataset.DatasetUtil(mnist_training).channel == 1
    assert len(dataset.sub_dataset(mnist_training, [1])) == 1
    cifar10 = DatasetCollection.get_by_name("CIFAR10")
    cifar10_training = cifar10.get_dataset(MachineLearningPhase.Training)

    assert dataset.DatasetUtil(cifar10_training).channel == 3
    assert dataset.DatasetUtil(cifar10_training).get_label_number() == 10
