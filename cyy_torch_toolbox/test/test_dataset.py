#!/usr/bin/env python3

from cyy_torch_toolbox.dataset import subset_dp
from cyy_torch_toolbox.dataset_collection import (
    ClassificationDatasetCollection, create_dataset_collection)
from cyy_torch_toolbox.dependency import (has_hugging_face,
                                          has_torch_geometric, has_torchtext,
                                          has_torchvision)
from cyy_torch_toolbox.ml_type import MachineLearningPhase


def test_dataset():
    if has_torchvision:
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
    if has_torch_geometric:
        dc = create_dataset_collection(
            ClassificationDatasetCollection,
            "KarateClub",
        )
    if has_hugging_face:
        dc = create_dataset_collection(
            ClassificationDatasetCollection,
            "multi_nli",
            dataset_kwargs={"val_split": "validation_matched"},
        )


def test_dataset_labels():
    if has_torchvision:
        for name in ("MNIST", "CIFAR10"):
            dc = create_dataset_collection(ClassificationDatasetCollection, name)
            assert len(dc.get_labels()) == 10
    if has_torchtext:
        for name in ("IMDB",):
            dc = create_dataset_collection(ClassificationDatasetCollection, name)
            assert dc.get_labels()
