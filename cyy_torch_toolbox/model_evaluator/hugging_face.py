import copy

import torch
import transformers
from cyy_naive_lib.reflection import get_kwarg_names

from .text import TextModelEvaluator


class HuggingFaceModelEvaluator(TextModelEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.model, transformers.modeling_utils.PreTrainedModel)

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
        if hasattr(self.model, "forward"):
            kwarg_names = get_kwarg_names(self.model.forward)
        else:
            kwarg_names = get_kwarg_names(self.model)
        if "input_ids" in kwarg_names and "inputs_embeds" in kwarg_names:
            if input_features is not None:
                if inputs is not None:
                    new_inputs = copy.copy(inputs)
                    new_inputs.pop("input_ids", None)
                else:
                    new_inputs = {}
                new_inputs["inputs_embeds"] = input_features
                output = self.model(**new_inputs, labels=targets)
            else:
                output = self.model(**inputs, labels=targets)
            return {
                "loss": output["loss"],
                "classification_output": output["logits"],
            }
        raise NotImplementedError()
