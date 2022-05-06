from transformers import AutoTokenizer

from .tokenizer import Tokenizer


def get_tokenizer(tokenizer_kwargs, dc=None):
    tokenizer_type = tokenizer_kwargs.pop("tokenizer_type", "spacy")
    if tokenizer_type == "spacy":
        return Tokenizer(dc, **tokenizer_kwargs)
    return AutoTokenizer.from_pretrained(tokenizer_type)
