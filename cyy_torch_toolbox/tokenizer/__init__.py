from collections.abc import Mapping
from typing import Any

import torch

type TokenIDType = int | tuple[int] | list[int] | torch.Tensor
type TokenIDsType = torch.Tensor


class Tokenizer:
    def get_vocab(self) -> Mapping[str, int]:
        raise NotImplementedError()

    def get_mask_token(self) -> str:
        raise NotImplementedError()

    def tokenize(self, phrase: str) -> list[str]:
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
