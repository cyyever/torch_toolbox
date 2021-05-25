from collections import Counter

import torch
from dataset import DatasetMapper
from torchtext.data.utils import get_tokenizer
from torchtext.legacy import data, datasets
from torchtext.vocab import Vocab


def get_text_and_label_fields():
    text_field = data.Field(tokenize="spacy", tokenizer_language="en_core_web_sm")
    # label_field = data.LabelField(dtype=torch.float)
    label_field = data.LabelField()
    return (text_field, label_field)
