from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ..dataset import DatasetCollection
from ..ml_type import MachineLearningPhase

type TokenIDType = int | Sequence[int] | torch.Tensor
type TokenIDsType = torch.Tensor


class TokenizerMixin:
    def get_token_id(self, token: str) -> TokenIDType:
        raise NotImplementedError()

    def get_tokens_from_transformed_result(self, transformed_result: Any) -> list[str]:
        raise NotImplementedError()

    def get_token(self, token_id: TokenIDType) -> str:
        raise NotImplementedError()

    def split_batch_input(self, inputs: Any, batch_size: int) -> dict:
        raise NotImplementedError()


class Tokenizer:
    def get_vocab(self) -> Mapping[str, int]:
        raise NotImplementedError()

    def tokenize(self, phrase: str) -> list[str] | Any:
        raise NotImplementedError()

    def get_token_id(self, token: str) -> TokenIDType:
        raise NotImplementedError()

    def get_tokens_from_transformed_result(self, transformed_result: Any) -> list[str]:
        raise NotImplementedError()

    def get_token(self, token_id: TokenIDType) -> str:
        raise NotImplementedError()

    def get_phrase(self, token_ids: TokenIDsType) -> str:
        return " ".join(self.get_token(token_id) for token_id in token_ids)

    def strip_special_tokens(self, token_ids: TokenIDsType) -> TokenIDsType:
        raise NotImplementedError()

    def split_batch_input(self, inputs: Any, batch_size: int) -> dict:
        raise NotImplementedError()


def collect_tokens(
    tokenizer: Tokenizer,
    dc: DatasetCollection,
    phase: MachineLearningPhase | None = None,
) -> Counter:
    counter: Counter = Counter()
    if phase is None:
        util_list = [
            dc.get_dataset_util(phase=phase)
            for phase in MachineLearningPhase
            if dc.has_dataset(phase)
        ]
    else:
        util_list = [dc.get_dataset_util(phase=phase)]
    for util in util_list:
        for index in range(len(util)):
            tokens = tokenizer.get_tokens_from_transformed_result(
                util._get_sample_input(index)
            )
            counter.update(tokens)
    return counter
