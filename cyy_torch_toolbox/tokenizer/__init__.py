from typing import Any

from ..dependency import has_hugging_face, has_spacy

if has_spacy:
    from .spacy import SpacyTokenizer

if has_hugging_face:
    import transformers


def get_tokenizer(dc, tokenizer_config: dict) -> Any:
    tokenizer_type: str = tokenizer_config.get("type", "spacy")
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
