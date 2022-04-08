#!/usr/bin/env python3

from cyy_torch_toolbox.dataset_collection import (
    ClassificationDatasetCollection, create_dataset_collection)


def test_tokenizer():
    imdb = create_dataset_collection(ClassificationDatasetCollection, "IMDB")
