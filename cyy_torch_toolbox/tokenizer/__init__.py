from typing import Any

from ..dependency import has_hugging_face, has_spacy, has_torchtext

if has_spacy and has_torchtext:
    from .spacy import SpacyTokenizer


if has_hugging_face:
    import transformers


def get_tokenizer(dc, tokenizer_config: dict | None = None) -> Any:
    if tokenizer_config is None:
        tokenizer_config = {}
    tokenizer_type = tokenizer_config.get("type", "spacy")
    match tokenizer_type:
        case "hugging_face":
            assert has_hugging_face
            return transformers.AutoTokenizer.from_pretrained(
                tokenizer_config["name"], **tokenizer_config.get("kwargs", {})
            )
        case "spacy":
            assert has_spacy
            return SpacyTokenizer(dc, **tokenizer_config.get("kwargs", {}))
    return None
