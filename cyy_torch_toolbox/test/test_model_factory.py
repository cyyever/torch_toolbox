#!/usr/bin/env python3
from cyy_torch_toolbox.dataset_collection import (
    ClassificationDatasetCollection, create_dataset_collection)
from cyy_torch_toolbox.model_factory import get_model


def test_model():
    # for dataset_name, model_name in [("IMDB", "aaa"), ("MNIST", "LeNet5")]:
    for dataset_name, model_name in [
        ("MNIST", "LeNet5"),
        ("MNIST", "efficientnet_b0"),
        # ("IMDB", "simplelstm"),
    ]:
        dc = create_dataset_collection(ClassificationDatasetCollection, dataset_name)
        get_model(model_name, dc, pretrained=False)
