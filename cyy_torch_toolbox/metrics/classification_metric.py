import torch

from ..ml_type import ModelType
from .metric import Metric


class ClassificationMetric(Metric):
    def _before_execution(self, **kwargs) -> None:
        executor = kwargs["executor"]
        if executor.running_model_evaluator.model_type != ModelType.Classification:
            self.disable()

    def _get_output(self, result: dict) -> torch.Tensor:
        output = result["model_output"]
        logits = result.get("logits", None)
        if logits is not None:
            output = logits
        if output.shape == 2:
            output = torch.max(output, dim=1).review(-1)
        if (output < 0).sum().item():
            output = output.sigmoid()
        return output
