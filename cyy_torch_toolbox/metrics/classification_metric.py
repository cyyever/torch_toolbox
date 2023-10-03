import torch

from ..ml_type import ModelType
from .metric import Metric


class ClassificationMetric(Metric):
    def _before_execution(self, **kwargs) -> None:
        executor = kwargs["executor"]
        if executor.running_model_evaluator.model_type != ModelType.Classification:
            self.disable()

    def _get_output(self, result: dict) -> torch.Tensor:
        output = result.get("logits", None)
        if output is None:
            output = result["model_output"]
        assert isinstance(output, torch.Tensor)
        if output.shape == 2:
            output = output.max(dim=1).view(-1)
        if (output < 0).sum().item():
            output = output.sigmoid()
        return output
