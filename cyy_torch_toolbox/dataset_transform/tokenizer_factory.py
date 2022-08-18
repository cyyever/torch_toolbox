from transformers import AutoTokenizer

from .tokenizer import SpacyTokenizer


def get_spacy_tokenizer(dc, **spacy_kwargs) -> SpacyTokenizer:
    return SpacyTokenizer(dc, **spacy_kwargs)


def get_hugging_face_tokenizer(tokenizer_type):
    return AutoTokenizer.from_pretrained(tokenizer_type)
