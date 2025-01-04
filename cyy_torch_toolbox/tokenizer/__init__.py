from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from cyy_naive_lib.log import log_error

from ..dataset import DatasetCollection
from ..executor import Executor
from ..ml_type import MachineLearningPhase

type TokenIDType = int | Sequence[int] | torch.Tensor
type TokenIDsType = torch.Tensor


class Tokenizer:
    def get_vocab(self) -> Mapping[str, int]:
        raise NotImplementedError()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def get_mask_token(self) -> str:
        raise NotImplementedError()

    def tokenize(self, phrase: str) -> list[str] | Any:
        raise NotImplementedError()

    def get_token_id(self, token: str) -> TokenIDType:
        raise NotImplementedError()

    def get_token_ids_from_transformed_result(
        self, transformed_result: Any
    ) -> TokenIDsType:
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


def convert_phrase_to_token_ids(
    executor: Executor,
    phrase: str,
    strip_special_token: bool = True,
) -> TokenIDsType:
    tokenizer = getattr(executor.model_evaluator, "tokenizer", None)
    assert isinstance(tokenizer, Tokenizer)
    transformed_token_results = tokenizer(phrase)
    token_ids = tokenizer.get_token_ids_from_transformed_result(
        transformed_token_results
    )
    if strip_special_token:
        token_ids = tokenizer.strip_special_tokens(token_ids)
        decoded_phrase = tokenizer.get_phrase(token_ids)
        if decoded_phrase.replace(" ", "") != phrase.replace(" ", ""):
            log_error("failed to recover phrase")
            log_error("phrase is: %s", phrase)
            log_error("decoded phrase is: %s", decoded_phrase)
            raise RuntimeError("failed to recover phrase")
    return token_ids


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
            transformed_token_results = util._get_sample_input(index)
            tokens = tokenizer.get_tokens_from_transformed_result(
                transformed_token_results
            )
            counter.update(tokens)
    return counter
