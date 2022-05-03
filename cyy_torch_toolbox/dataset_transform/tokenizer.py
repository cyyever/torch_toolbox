from collections import Counter, OrderedDict

import spacy
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from torchtext.vocab import Vocab, build_vocab_from_iterator


class Tokenizer:
    def __init__(
        self,
        dc,
        special_tokens: None | list[str] = None,
        keep_punct: bool = True,
        keep_stop: bool = True,
        min_freq: int = 1,
        max_tokens=None,
        **kwargs
    ):
        self.__keep_punct = keep_punct
        self.__keep_stop = keep_stop
        self.__spacy = spacy.load("en_core_web_sm")
        self.unusual_words: set = set()

        counter: Counter = Counter()

        def yield_tokens():
            nonlocal counter
            for phase in MachineLearningPhase:
                dataset = dc.get_dataset(phase=phase)
                for data in dataset:
                    data = dc.get_transforms(phase=phase).extract_data(data)
                    text = data["input"]
                    text = dc.get_transforms(phase=phase).transform_text(text)
                    tokens = self.__tokenize(text)
                    counter.update(tokens)
                    yield tokens

        if special_tokens is None:
            special_tokens = []
        for token in ("<pad>", "<unk>", "<mask>"):
            if token not in special_tokens:
                special_tokens.append(token)
        for token in special_tokens:
            self.__spacy.tokenizer.add_special_case(
                token, [{spacy.symbols.ORTH: token}]
            )
        vocab = build_vocab_from_iterator(
            yield_tokens(),
            specials=special_tokens,
            min_freq=min_freq,
            max_tokens=max_tokens,
            **kwargs
        )

        # First sort by descending frequency, then lexicographically
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        if max_tokens is None:
            freq_dict = OrderedDict(sorted_by_freq_tuples)
        else:
            freq_dict = OrderedDict(
                sorted_by_freq_tuples[: max_tokens - len(special_tokens)]
            )

        vocab.set_default_index(vocab["<unk>"])
        get_logger().info("vocab size is %s", len(vocab))
        self.__vocab: Vocab = vocab
        self.__freq_dict = freq_dict

    @property
    def freq_dict(self) -> OrderedDict:
        return self.__freq_dict

    @property
    def vocab(self) -> Vocab:
        return self.__vocab

    def spacy_model(self):
        return self.__spacy

    def __tokenize(self, s):
        tokens = self.__spacy.tokenizer(s)
        if not self.__keep_punct:
            tokens = [t for t in tokens if not t.is_punct]
        if not self.__keep_stop:
            tokens = [t for t in tokens if not t.is_stop]
        return [t.text for t in tokens]

    def __call__(self, s):
        return self.__vocab(self.__tokenize(s))
