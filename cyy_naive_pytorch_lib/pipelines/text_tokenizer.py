from collections import Counter

from dataset import DatasetMapper
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab


class TextTokenizer(DatasetMapper):
    def __init__(self, dataset, lang="basic_english"):
        super().__init__(dataset=dataset, mappers=[])
        self.__tokenizer = get_tokenizer(lang)
        counter = Counter()
        for (label, line) in self.dataset:
            counter.update(self.__tokenizer(line))
        self.__vocab = Vocab(counter, min_freq=1)
        self.add_mapper(
            lambda x: [self.__vocab[token] for token in self.__tokenizer(x)]
        )
