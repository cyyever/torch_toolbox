from collections import Counter, OrderedDict

from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab, build_vocab_from_iterator


class Tokenizer:
    def __init__(
        self, dc, special_tokens: None | list[str] = None, min_freq: int = 1, **kwargs
    ):
        tokenizer = get_tokenizer(tokenizer="spacy", language="en_core_web_sm")

        counter = Counter()

        def yield_tokens():
            nonlocal counter
            for phase in MachineLearningPhase:
                dataset = dc.get_dataset(phase=phase)
                for text, _ in dataset:
                    text = dc.get_transforms(phase=phase).transform_text(text)
                    tokens = tokenizer(text)
                    counter.update(tokens)
                    yield tokens

        if special_tokens is None:
            special_tokens = []
        for token in ("<pad>", "<unk>", "<mask>"):
            if token not in special_tokens:
                special_tokens.append(token)
        self.__tokenizer = tokenizer
        vocab = build_vocab_from_iterator(
            yield_tokens(), specials=special_tokens, min_freq=min_freq, **kwargs
        )

        max_tokens = kwargs.get("max_tokens", None)
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
    def tokenizer(self):
        return self.__tokenizer

    @property
    def freq_dict(self) -> OrderedDict:
        return self.__freq_dict

    @property
    def vocab(self) -> Vocab:
        return self.__vocab

    def __call__(self, s):
        return self.__vocab(self.__tokenizer(s))
