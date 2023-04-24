#!/usr/bin/env python3

from cyy_torch_toolbox.dataset_collection import (
    ClassificationDatasetCollection, create_dataset_collection)
from cyy_torch_toolbox.dependency import has_torchtext


def test_tokenizer():
    if has_torchtext:
        imdb = create_dataset_collection(ClassificationDatasetCollection, "IMDB")
