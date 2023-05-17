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

    def get_input_feature(self, inputs):
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

    def _forward_model(
        self, inputs, targets, input_features=None, **kwargs
    ) -> dict | torch.Tensor:
        if input_features is not None:
            if inputs is not None:
                inputs = copy.copy(inputs)
                inputs.pop("input_ids", None)
            else:
                inputs = {}
            inputs["inputs_embeds"] = input_features

        if hasattr(targets, "input_ids"):
            targets = targets.input_ids
        # get_logger().error("inputs %s,labels %s", inputs, targets)
        output = self.model(**inputs, labels=targets)
        return {
            "model_output": output,
            "logits": output.logits,
            "loss": output.loss,
            "is_averaged_loss": True,
        }
