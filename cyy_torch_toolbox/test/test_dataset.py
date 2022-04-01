#!/usr/bin/env python3

from cyy_torch_toolbox.dataset import sub_dataset
from cyy_torch_toolbox.dataset_util import DatasetUtil
from cyy_torch_toolbox.dataset_collection import DatasetCollection
from cyy_torch_toolbox.ml_type import MachineLearningPhase


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
    assert next(mnist.generate_raw_data(MachineLearningPhase.Training))
    assert DatasetUtil(mnist_training).channel == 1
    assert len(sub_dataset(mnist_training, [1])) == 1
    cifar10 = DatasetCollection.get_by_name("CIFAR10")
    cifar10_training = cifar10.get_dataset(MachineLearningPhase.Training)
    assert DatasetUtil(cifar10_training).channel == 3
    assert DatasetUtil(cifar10_training).get_label_number() == 10
    assert abs(
        len(cifar10.get_dataset(MachineLearningPhase.Test))
        - len(cifar10.get_dataset(MachineLearningPhase.Validation))
        <= 1
    )
    print("cifar10 labels are", cifar10.get_label_names())


def test_dataset_labels():
    for name in ("MNIST", "CIFAR10"):
        dc = DatasetCollection.get_by_name(name)
        assert len(dc.get_labels()) == 10
    for name in ("IMDB",):
        dc = DatasetCollection.get_by_name(name)
        assert dc.get_labels()
