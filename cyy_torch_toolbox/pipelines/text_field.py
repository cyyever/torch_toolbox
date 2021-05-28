from torchtext.legacy import data


def get_text_and_label_fields():
    text_field = data.Field(
        tokenize="spacy", tokenizer_language="en_core_web_sm", include_lengths=True
    )
    label_field = data.LabelField()
    return (text_field, label_field)
