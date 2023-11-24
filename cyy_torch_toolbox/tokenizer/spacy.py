import functools
from collections import Counter, OrderedDict
from typing import Iterable

from cyy_naive_lib.log import get_logger
from torchtext.vocab import Vocab, vocab

import spacy
from spacy.symbols import ORTH

from .util import collect_tokens


class SpacyTokenizer:
    def __init__(
        self,
        dc,
        package_name: str = "en_core_web_sm",
        special_tokens: None | Iterable[str] | set[str] = None,
        keep_punct: bool = True,
        keep_stop: bool = True,
        min_freq: int = 1,
        max_tokens: None | int = None,
    ) -> None:
        self.__keep_punct = keep_punct
        self.__keep_stop = keep_stop
        self.__spacy = spacy.load(package_name)
        self.unusual_words: set = set()

        counter: Counter = dc.get_cached_data(
            file="tokenizer_word_counter.pk",
            computation_fun=functools.partial(
                collect_tokens, dc=dc, tokenizer=self.__spacy.tokenizer
            ),
        )

        if special_tokens is None:
            special_tokens = set()
        else:
            special_tokens = set(special_tokens)
        for token in ("<pad>", "<unk>", "<mask>", "<cls>", "<sep>"):
            special_tokens.add(token)
        for token in special_tokens:
            self.__spacy.tokenizer.add_special_case(token, [{ORTH: token}])

        # First sort by descending frequency, then lexicographically
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        if max_tokens is None:
            ordered_dict = OrderedDict(sorted_by_freq_tuples)
        else:
            assert (
                len(special_tokens) < max_tokens
            ), "len(special_tokens) >= max_tokens, so the vocab will be entirely special tokens."
            ordered_dict = OrderedDict(
                sorted_by_freq_tuples[: max_tokens - len(special_tokens)]
            )

        word_vocab = vocab(
            ordered_dict,
            min_freq=min_freq,
            specials=list(special_tokens),
        )

        word_vocab.set_default_index(word_vocab["<unk>"])
        get_logger().info("vocab size is %s", len(word_vocab))
        self.__vocab: Vocab = word_vocab
        self.__freq_dict = ordered_dict

    @property
    def freq_dict(self) -> OrderedDict:
        return self.__freq_dict

    @property
    def vocab(self) -> Vocab:
        return self.__vocab

    def spacy_model(self):
        return self.__spacy

    def __tokenize(self, s) -> list[str]:
        tokens = self.__spacy.tokenizer(s)
        return [
            t.text
            for t in tokens
            if (self.__keep_punct or not t.is_punct)
            and (self.__keep_stop or not t.is_stop)
        ]

    def __call__(self, s) -> list[int]:
        return self.__vocab(self.__tokenize(s))
