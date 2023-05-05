import copy

import torch
import transformers
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.reflection import get_kwarg_names

from ..ml_type import ModelType
from .text import TextModelEvaluator


class HuggingFaceModelEvaluator(TextModelEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.model, transformers.modeling_utils.PreTrainedModel)
        assert self.model_type == ModelType.Classification

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
        get_logger().error("output is %s %s", output.loss,output.logits)
        return {
            "loss": output["loss"],
            "classification_output": output["logits"],
        }
