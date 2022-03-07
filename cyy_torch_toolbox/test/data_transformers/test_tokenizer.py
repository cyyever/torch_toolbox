#!/usr/bin/env python3

from cyy_torch_toolbox.dataset_collection import DatasetCollection
from cyy_torch_toolbox.dataset_transformers.tokenizer import Tokenizer


def test_tokenizer():
    imdb = DatasetCollection.get_by_name("IMDB")
    tokenizer = Tokenizer(imdb)
