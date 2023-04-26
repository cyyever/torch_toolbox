from ..dependency import has_hugging_face, has_spacy

if has_spacy:
    from .spacy import SpacyTokenizer


def get_spacy_tokenizer(dc, **spacy_kwargs: dict) -> SpacyTokenizer:
    if not has_spacy:
        raise RuntimeError("no spacy library")
    return SpacyTokenizer(dc, **spacy_kwargs)


if has_hugging_face:
    from transformers import AutoTokenizer


def get_hugging_face_tokenizer(tokenizer_type):
    if not has_hugging_face:
        raise RuntimeError("no hugging face library")
    return AutoTokenizer.from_pretrained(tokenizer_type)
