from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class Tokenizer:
    def __init__(self, dc, special_tokens: None | list[str] = None, min_freq=1):
        tokenizer = get_tokenizer(tokenizer="spacy", language="en_core_web_sm")

        def yield_tokens():
            for dataset in dc.foreach_dataset():
                for text, target in dataset:
                    yield tokenizer(text)
                    yield tokenizer(target)

        if special_tokens is None:
            special_tokens = []
        if "<pad>" not in special_tokens:
            special_tokens.append("<pad>")
        vocab = build_vocab_from_iterator(
            yield_tokens(), specials=special_tokens, min_freq=min_freq
        )
        if special_tokens is not None and "<unk>" in special_tokens:
            vocab.set_default_index(vocab["<unk>"])
        self.__tokenizer = tokenizer
        self.__vocab = vocab

    @property
    def tokenizer(self):
        return self.__tokenizer

    @property
    def vocab(self):
        return self.__vocab

    def __call__(self, s):
        return self.__vocab(self.__tokenizer(s))
