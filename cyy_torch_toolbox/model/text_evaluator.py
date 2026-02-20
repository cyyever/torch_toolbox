from typing import Any

from ..tokenizer import Tokenizer


class TextModelEvaluatorMixin:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer: Tokenizer = tokenizer

    def split_batch_input(self, inputs: Any, **kwargs: Any) -> dict[str, Any]:
        return self.tokenizer.split_batch_input(inputs, **kwargs)
