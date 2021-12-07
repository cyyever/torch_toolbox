from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class TokenizerAndVocab:
    def __init__(self, dataset, lang="en_core_web_sm"):
        tokenizer = get_tokenizer(tokenizer="spacy", language=lang)

        def yield_tokens(data_iter):
            for _, text in data_iter:
                yield tokenizer(text)

        vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        self.__tokenizer = tokenizer
        self.__vocab = vocab

    def __call__(self, s):
        return self.__vocab(self.__tokenizer(s))
