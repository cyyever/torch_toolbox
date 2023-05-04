from typing import Any

from ..dependency import has_hugging_face, has_spacy, has_torchtext

if has_spacy and has_torchtext:
    from .spacy import SpacyTokenizer


if has_hugging_face:
    import transformers


def __get_hugging_face_tokenizer(tokenizer_type):
    if not has_hugging_face:
        raise RuntimeError("no hugging face library")
    return transformers.AutoTokenizer.from_pretrained(tokenizer_type)


def get_tokenizer(dc, dataset_kwargs: dict | None) -> Any:
    if not dataset_kwargs:
        dataset_kwargs = {}
    tokenizer_config = dataset_kwargs.get("tokenizer", {})
    tokenizer_type = tokenizer_config.get("type", "spacy")
    match tokenizer_type:
        case "hugging_face":
            assert has_hugging_face
            return __get_hugging_face_tokenizer(
                dataset_kwargs.get("tokenizer", {})["name"]
            )
        case "spacy":
            assert has_spacy
            return SpacyTokenizer(dc, **tokenizer_config.get("kwargs", {}))
    return None
