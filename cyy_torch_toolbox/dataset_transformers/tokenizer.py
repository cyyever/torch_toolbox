from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class Tokenizer:
    def __init__(self, dc, lang="en_core_web_sm", special_tokens=None, min_freq=1):
        tokenizer = get_tokenizer(tokenizer="spacy", language=lang)

        def yield_tokens():
            for dataset in dc.foreach_dataset():
                for target, text in dataset:
                    print(target)
                    print(text)
                    yield tokenizer(text)
                    yield tokenizer(target)

        if special_tokens is None:
            special_tokens = {"<pad>", "<unk>"}

        vocab = build_vocab_from_iterator(
            yield_tokens(), specials=list(special_tokens), min_freq=min_freq
        )
        if "<unk>" in special_tokens:
            vocab.set_default_index(vocab["<unk>"])
        self.__tokenizer = tokenizer
        self.__vocab = vocab

    @property
    def vocab(self):
        return self.__vocab

    def __call__(self, s):
        return self.__vocab(self.__tokenizer(s))
