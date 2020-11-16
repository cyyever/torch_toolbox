#!/usr/bin/env python3

import torch
import dataset


def test_dataset():
    mnist_dataset = dataset.get_dataset("MNIST", dataset.DatasetType.Training)
    mnist_validation_dataset = dataset.get_dataset(
        "MNIST", dataset.DatasetType.Validation
    )
    assert abs(
        len(mnist_dataset) /
        len(mnist_validation_dataset) -
        5) < 0.01
    dataset.sub_dataset(mnist_dataset, [1])
    assert dataset.DatasetUtil(mnist_dataset).channel == 1
    cifar10_dataset = dataset.get_dataset(
        "CIFAR10", dataset.DatasetType.Training)
    assert dataset.DatasetUtil(cifar10_dataset).channel == 3
    assert dataset.DatasetUtil(mnist_dataset).get_label_number() == 10
    assert dataset.DatasetUtil(cifar10_dataset).get_label_number() == 10
