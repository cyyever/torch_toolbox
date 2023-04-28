import functools

import transformers
from cyy_naive_lib.log import get_logger

from ..ml_type import DatasetType

__huggingface_models = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-cased",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "wietsedv/bert-base-dutch-cased",
    "openai-gpt",
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "transfo-xl-wt103",
    "xlnet-base-cased",
    "xlnet-large-cased",
    "xlm-mlm-en-2048",
    "xlm-mlm-ende-1024",
    "xlm-mlm-enfr-1024",
    "xlm-mlm-enro-1024",
    "xlm-mlm-xnli15-1024",
    "xlm-mlm-tlm-xnli15-1024",
    "xlm-clm-enfr-1024",
    "xlm-clm-ende-1024",
    "xlm-mlm-17-1280",
    "xlm-mlm-100-1280",
    "roberta-base",
    "roberta-large",
    "roberta-large-mnli",
    "roberta-large",
    "distilroberta-base",
    "roberta-base-openai-detector",
    "roberta-base",
    "roberta-large-openai-detector",
    "roberta-large",
    "distilbert-base-uncased",
    "distilbert-base-uncased-distilled-squad",
    "distilbert-base-cased",
    "distilbert-base-cased-distilled-squad",
    "distilgpt2",
    "distilbert-base-german-cased",
    "distilbert-base-multilingual-cased",
    "ctrl",
    "camembert-base",
    "albert-base-v1",
    "albert-large-v1",
    "albert-xlarge-v1",
    "albert-xxlarge-v1",
    "albert-base-v2",
    "albert-large-v2",
    "albert-xlarge-v2",
    "albert-xxlarge-v2",
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3B",
    "t5-11B",
    "xlm-roberta-base",
    "xlm-roberta-large",
    "flaubert/flaubert_small_cased",
    "flaubert/flaubert_base_uncased",
    "flaubert/flaubert_base_cased",
    "flaubert/flaubert_large_cased",
    "facebook/bart-large",
    "facebook/bart-base",
    "facebook/bart-large-mnli",
    "facebook/bart-large-cnn",
    "DialoGPT-small",
    "DialoGPT-medium",
    "DialoGPT-large",
    "reformer-enwik8",
    "reformer-crime-and-punishment",
    "Helsinki-NLP/opus-mt-{src}-{tgt}",
    "google/pegasus-{dataset}",
    "allenai/longformer-base-4096",
    "allenai/longformer-large-4096",
    "facebook/mbart-large-cc25",
    "facebook/mbart-large-en-ro",
    "lxmert-base-uncased",
    "funnel-transformer/small",
    "funnel-transformer/small-base",
    "funnel-transformer/medium",
    "funnel-transformer/medium-base",
    "funnel-transformer/intermediate",
    "funnel-transformer/intermediate-base",
    "funnel-transformer/large",
    "funnel-transformer/large-base",
    "funnel-transformer/xlarge",
    "funnel-transformer/xlarge-base",
    "microsoft/layoutlm-base-uncased",
    "microsoft/layoutlm-large-uncased",
]


def __create_hugging_face_classification_model(model_name, pretrained, **model_kwargs):
    pretrained_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name, **model_kwargs
    )
    if pretrained:
        return pretrained_model
    get_logger().warning("use huggingface without pretrained parameters")
    old_embedding = pretrained_model.get_input_embeddings()
    config = transformers.AutoConfig.from_pretrained(model_name, **model_kwargs)
    model = transformers.AutoModelForSequenceClassification.from_config(config)
    model.set_input_embeddings(old_embedding)
    return model


def __create_hugging_face_model(model_name, pretrained, **model_kwargs):
    pretrained_model = transformers.AutoModel.from_pretrained(
        model_name, **model_kwargs
    )
    if pretrained:
        return pretrained_model
    get_logger().warning("use huggingface without pretrained parameters")
    # old_embedding = pretrained_model.get_input_embeddings()
    config = transformers.AutoConfig.from_pretrained(model_name, **model_kwargs)
    model = transformers.AutoModel.from_config(config)
    # model.set_input_embeddings(old_embedding)
    return model


def get_hugging_face_model_info() -> dict:
    model_info: dict = {}
    for model_name in __huggingface_models:
        full_model_name = "hugging_face_sequence_classification_" + model_name
        model_info[full_model_name.lower()] = {
            "name": full_model_name,
            "constructor": functools.partial(
                __create_hugging_face_classification_model,
                model_name,
            ),
            "has_tokenizer": True,
        }
        full_model_name = "hugging_face_" + model_name
        model_info[full_model_name.lower()] = {
            "name": full_model_name,
            "constructor": functools.partial(
                __create_hugging_face_model,
                model_name,
            ),
            "has_tokenizer": True,
        }
    return {DatasetType.Text: model_info}
