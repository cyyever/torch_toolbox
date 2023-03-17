#!/usr/bin/env python3

from cyy_torch_toolbox.dataset import subset_dp
from cyy_torch_toolbox.dataset_collection import (
    ClassificationDatasetCollection, create_dataset_collection)
from cyy_torch_toolbox.ml_type import MachineLearningPhase


def test_dataset():
    mnist = create_dataset_collection(ClassificationDatasetCollection, "MNIST")
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
    assert mnist.get_dataset_util().channel == 1
    assert len(subset_dp(mnist_training, [1])) == 1
    cifar10 = create_dataset_collection(ClassificationDatasetCollection, "CIFAR10")
    assert cifar10.get_dataset_util().channel == 3
    assert len(cifar10.get_dataset_util().get_labels()) == 10
    assert abs(
        len(cifar10.get_dataset(MachineLearningPhase.Test))
        - len(cifar10.get_dataset(MachineLearningPhase.Validation))
        <= 1
    )
    print("cifar10 labels are", cifar10.get_label_names())


def test_dataset_labels():
    for name in ("MNIST", "CIFAR10"):
        dc = create_dataset_collection(ClassificationDatasetCollection, name)
        assert len(dc.get_labels()) == 10
    for name in ("IMDB",):
        dc = create_dataset_collection(ClassificationDatasetCollection, name)
        assert dc.get_labels()


def test_torch_geometric_dataset():
    dc = create_dataset_collection(
        ClassificationDatasetCollection,
        "KarateClub",
    )


def test_hugging_face_dataset():
    return
    dc = create_dataset_collection(
        ClassificationDatasetCollection,
        "multi_nli",
        dataset_kwargs={"val_split": "validation_matched"},
    )
