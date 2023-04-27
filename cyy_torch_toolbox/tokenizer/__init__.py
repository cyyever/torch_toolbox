from ..dependency import has_hugging_face, has_spacy

if has_spacy:
    from .spacy import SpacyTokenizer


if has_hugging_face:
    import transformers


def get_hugging_face_tokenizer(tokenizer_type):
    if not has_hugging_face:
        raise RuntimeError("no hugging face library")
    return transformers.AutoTokenizer.from_pretrained(tokenizer_type)


def get_tokenizer(dc, dataset_kwargs: dict, model_evaluator=None):
    tokenizer = None
    if (
        has_hugging_face
        and model_evaluator is not None
        and isinstance(
            model_evaluator.get_underlying_model(),
            transformers.modeling_utils.PreTrainedModel,
        )
    ):
        tokenizer = get_hugging_face_tokenizer(
            model_evaluator.model_name.replace("sequence_classification_", "")
        )
    if tokenizer is None and has_spacy:
        tokenizer_kwargs = dataset_kwargs.get("tokenizer", {})
        spacy_kwargs = tokenizer_kwargs.get("spacy_kwargs", {})
        tokenizer = SpacyTokenizer(dc, **spacy_kwargs)
    return tokenizer
