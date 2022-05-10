from transformers import AutoTokenizer

from .tokenizer import SpacyTokenizer


def get_tokenizer(tokenizer_kwargs, dc=None):
    tokenizer_type = tokenizer_kwargs.pop("type", "spacy")
    if tokenizer_type == "spacy":
        return SpacyTokenizer(dc, **tokenizer_kwargs)
    return AutoTokenizer.from_pretrained(tokenizer_type, **tokenizer_kwargs)
