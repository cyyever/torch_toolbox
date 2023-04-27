from collections import Counter, OrderedDict

from cyy_naive_lib.log import get_logger
from torchtext.vocab import Vocab, vocab

import spacy

from ..dataset_collection import DatasetCollection
from ..ml_type import MachineLearningPhase


class SpacyTokenizer:
    def __init__(
        self,
        dc: DatasetCollection,
        special_tokens: None | list[str] = None,
        keep_punct: bool = True,
        keep_stop: bool = True,
        min_freq: int = 1,
        max_tokens: None | int = None,
    ) -> None:
        self.__keep_punct = keep_punct
        self.__keep_stop = keep_stop
        self.__spacy = spacy.load("en_core_web_sm")
        self.unusual_words: set = set()

        def computation_fun():
            nonlocal dc
            counter: Counter = Counter()

            for phase in MachineLearningPhase:
                if not dc.has_dataset(phase=phase):
                    continue
                dataset = dc.get_dataset(phase=phase)
                for data in dataset:
                    data = dc.get_transforms(phase=phase).extract_data(data)
                    input_data = data["input"]
                    match input_data:
                        case str():
                            elements = [input_data]
                        case [*elements]:
                            pass
                        case _:
                            raise NotImplementedError(type(input_data))
                    for text in elements:
                        text = dc.get_transforms(phase=phase).transform_text(text)
                        tokens = self.__tokenize(text)
                        counter.update(tokens)
            return counter

        counter: Counter = dc.get_cached_data(
            file="tokenizer_word_counter.pk", computation_fun=computation_fun
        )

        if special_tokens is None:
            special_tokens = []
        for token in ("<pad>", "<unk>", "<mask>", "<cls>", "<sep>"):
            if token not in special_tokens:
                special_tokens.append(token)
        for token in special_tokens:
            self.__spacy.tokenizer.add_special_case(
                token, [{spacy.symbols.ORTH: token}]
            )

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
            specials=special_tokens,
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

    def __tokenize(self, s):
        tokens = self.__spacy.tokenizer(s)
        return [
            t.text
            for t in tokens
            if (self.__keep_punct or not t.is_punct)
            and (self.__keep_stop or not t.is_stop)
        ]

    def __call__(self, s):
        return self.__vocab(self.__tokenize(s))
