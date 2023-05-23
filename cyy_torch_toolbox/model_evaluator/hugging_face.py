from typing import Any

import torch

from ..ml_type import ModelType
from .text import TextModelEvaluator


class HuggingFaceModelEvaluator(TextModelEvaluator):
    def __init__(self, model, model_type, **kwargs: Any) -> None:
        if model_type is None:
            if "ConditionalGeneration" in model.__class__.__name__:
                model_type = ModelType.TextGeneration
        super().__init__(model=model, model_type=model_type, **kwargs)

    def split_batch_input(self, inputs, targets) -> tuple:
        batch_dim = 0
        new_inputs = []
        first_value = next(iter(inputs.values()))
        assert isinstance(first_value, torch.Tensor)
        for i in range(first_value.size(dim=0)):
            new_inputs.append({k: v[i].unsqueeze(dim=0) for k, v in inputs.items()})
        inputs = new_inputs
        return inputs, batch_dim

    def get_input_feature(self, inputs) -> dict:
        input_ids = inputs["input_ids"]
        if hasattr(self.model, "distilbert"):
            if len(list(input_ids.shape)) == 1:
                input_ids = input_ids.unsqueeze(dim=0)
            embeddings = self.model.distilbert.embeddings(input_ids).detach()
        elif hasattr(self.model, "bert"):
            embeddings = self.model.get_input_embeddings()(input_ids).detach()
        else:
            raise NotImplementedError(self.model)
        inputs.pop("input_ids", None)
        inputs["inputs_embeds"] = embeddings
        return inputs

    def _create_input(
        self, inputs: dict, targets, is_input_feature: bool, **kwargs: Any
    ) -> dict:
        if hasattr(targets, "input_ids"):
            targets = targets.input_ids
        inputs["labels"] = targets
        return inputs

    def _forward_model(self, **kwargs: Any) -> dict:
        model_input = self._create_input(**kwargs)
        output = self.model(**model_input)
        return {
            "model_input": model_input,
            "model_output": output,
            "logits": output.logits,
            "loss": output.loss,
            "is_averaged_loss": True,
        }
