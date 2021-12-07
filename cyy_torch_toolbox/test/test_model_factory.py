#!/usr/bin/env python3
from dataset_collection import DatasetCollection
from model_factory import get_model


def test_model():
    # for dataset_name, model_name in [("IMDB", "aaa"), ("MNIST", "LeNet5")]:
    for dataset_name, model_name in [
        ("MNIST", "LeNet5"),
        ("MNIST", "efficientnet_b0"),
        ("IMDB", "simplelstm"),
    ]:
        dc = DatasetCollection.get_by_name(dataset_name)
        get_model(model_name, dc, pretrained=False)
