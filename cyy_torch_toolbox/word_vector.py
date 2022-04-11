import codecs
import os

import torch
import torch.nn as nn
from cyy_naive_lib.source_code.tarball_source import TarballSource
from torchtext.vocab import Vocab

from cyy_torch_toolbox.model_util import ModelUtil


class PretrainedWordVector:
    __word_vector_root_dir: str = os.path.join(
        os.path.expanduser("~"), "pytorch_word_vector"
    )

    def __init__(self, name: str):
        self.__word_vector_dict: dict = self.__download(name)

    @property
    def word_vector_dict(self):
        return self.__word_vector_dict

    def load_to_model(self, model_util: ModelUtil, vocab: Vocab) -> None:
        itos = vocab.get_itos()

        def __load_embedding(_, layer):
            embeddings = layer.weight.tolist()
            for idx, token in enumerate(itos):
                word_vector = self.__word_vector_dict.get(token, None)
                if word_vector is None:
                    word_vector = self.__word_vector_dict.get(token.lower(), None)
                if word_vector is not None:
                    embeddings[idx] = word_vector
            assert list(layer.weight.shape) == [
                len(itos),
                len(next(iter(self.__word_vector_dict.values()))),
            ], "Shape of weight does not match num_embeddings and embedding_dim"
            layer.weight = nn.Parameter(embeddings)

        model_util.change_sub_modules(nn.Embedding, __load_embedding)

    @classmethod
    def get_root_dir(cls) -> str:
        return os.getenv("pytorch_word_vector_root_dir", cls.__word_vector_root_dir)

    @classmethod
    def __download(cls, name: str) -> dict:
        word_vector_dict: dict = {}
        tarball = None
        if name == "glove.6B":
            tarball = TarballSource(
                spec=name,
                url="http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip",
                root_dir=cls.get_root_dir(),
            )
        if tarball is None:
            raise RuntimeError(f"unknown word vector {name}")
        with tarball:
            if name == "glove.6B":
                with codecs.open("glove.6B.300d.txt", "r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip().split()
                        word_vector_dict[s[0]] = torch.Tensor([float(i) for i in s[1:]])
        return word_vector_dict