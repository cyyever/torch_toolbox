import copy
from typing import Any

import torch
import transformers

from ..ml_type import ModelType
from .text import TextModelEvaluator


class HuggingFaceModelEvaluator(TextModelEvaluator):
    def __init__(self, model, model_type, **kwargs: Any) -> None:
        if model_type is None:
            if "ConditionalGeneration" in model.__class__.__name__:
                model_type = ModelType.TextGeneration
        super().__init__(model=model, model_type=model_type, **kwargs)

    def split_batch_input(self, inputs, targets, input_features=None) -> tuple:
        batch_dim = 0
        new_inputs = []
        first_value = next(iter(inputs.values()))
        assert isinstance(first_value, torch.Tensor)
        for i in range(first_value.size(dim=0)):
            new_inputs.append(
                {k: v[i].unsqueeze(dim=0) for k, v in inputs.items()}
            )
        inputs = new_inputs
        return inputs, batch_dim, input_features

    def get_input_feature(self, inputs) -> torch.Tensor:
        match inputs:
            case transformers.tokenization_utils_base.BatchEncoding() | dict():
                input_ids = inputs["input_ids"]
            case _:
                input_ids = inputs
        if hasattr(self.model, "distilbert"):
            if len(list(input_ids.shape)) == 1:
                input_ids = input_ids.unsqueeze(dim=0)
            return self.model.distilbert.embeddings(input_ids).detach()
        if hasattr(self.model, "bert"):
            return self.model.get_input_embeddings()(input_ids).detach()
        raise NotImplementedError(self.model)

    def _create_input(self, inputs: dict, targets, input_features=None) -> dict:
        if input_features is not None:
            if inputs is not None:
                inputs = copy.copy(inputs)
                inputs.pop("input_ids", None)
            else:
                inputs = {}
            inputs["inputs_embeds"] = input_features

        if hasattr(targets, "input_ids"):
            targets = targets.input_ids
        inputs["labels"] = targets
        return inputs

    def _forward_model(
        self, inputs: dict, targets, input_features=None, **kwargs: Any
    ) -> dict:
        model_input = self._create_input(
            inputs=inputs, targets=targets, input_features=input_features
        )
        output = self.model(**model_input)
        return {
            "model_input": model_input,
            "model_output": output,
            "logits": output.logits,
            "loss": output.loss,
            "is_averaged_loss": True,
        }
