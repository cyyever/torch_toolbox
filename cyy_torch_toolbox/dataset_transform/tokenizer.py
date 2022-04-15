from cyy_torch_toolbox.ml_type import MachineLearningPhase
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab, build_vocab_from_iterator


class Tokenizer:
    def __init__(self, dc, special_tokens: None | list[str] = None, min_freq: int = 1):
        tokenizer = get_tokenizer(tokenizer="spacy", language="en_core_web_sm")

        def yield_tokens():
            for phase in MachineLearningPhase:
                dataset = dc.get_dataset(phase=phase)
                for text, _ in dataset:
                    text = dc.get_transforms(phase=phase).transform_text(text)
                    yield tokenizer(text)

        if special_tokens is None:
            special_tokens = []
        for token in ("<pad>", "<unk>", "<mask>"):
            if token not in special_tokens:
                special_tokens.append(token)
        self.__tokenizer = tokenizer
        vocab = build_vocab_from_iterator(
            yield_tokens(), specials=special_tokens, min_freq=min_freq
        )
        vocab.set_default_index(vocab["<unk>"])
        self.__vocab: Vocab = vocab

    @property
    def tokenizer(self):
        return self.__tokenizer

    @property
    def vocab(self) -> Vocab:
        return self.__vocab

    def __call__(self, s):
        return self.__vocab(self.__tokenizer(s))
