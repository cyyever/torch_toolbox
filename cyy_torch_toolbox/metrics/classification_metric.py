from typing import Any, Literal

import torch
import torchmetrics.metric

from ..ml_type import ModelType
from .metric import Metric


class ClassificationMetric(Metric):
    _metric: None | torchmetrics.metric.Metric = None

    @property
    def metric(self) -> torchmetrics.metric.Metric:
        assert self._metric is not None
        return self._metric

    def _before_execution(self, **kwargs) -> None:
        executor = kwargs["executor"]
        if executor.running_model_evaluator.model_type != ModelType.Classification:
            self.disable()

    def _before_epoch(self, **kwargs: Any) -> None:
        self._metric = None

    def _get_output(self, result: dict) -> torch.Tensor:
        output = result.get("logits")
        if output is None:
            output = result["original_output"]
        assert isinstance(output, torch.Tensor)
        output = output.detach()
        output = torch.where(torch.any(output < 0), output.sigmoid(), output)
        if len(output.shape) == 2 and output.shape[1] == 1:
            output = torch.stack((1 - output, output), dim=2).squeeze()
        return output

    @torch.no_grad()
    def _get_task(self, executor) -> Literal["binary", "multiclass", "multilabel"]:
        if executor.dataset_collection.is_mutilabel():
            return "multilabel"
        if executor.dataset_collection.label_number <= 2:
            return "binary"
        return "multiclass"

    @torch.no_grad()
    def _get_new_output(
        self, executor, result: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        targets = result["targets"]
        output = result.get("logits")
        if output is None:
            output = result.get("original_output")
        assert isinstance(output, torch.Tensor)
        assert isinstance(targets, torch.Tensor)
        mask = targets != -100
        new_output = output[mask]
        targets = targets[mask]

        with executor.device:
            if (
                executor.dataset_collection.label_number <= 2
                and new_output.shape[-1] == 2
            ):
                new_output = torch.argmax(new_output, dim=-1)
        return new_output, targets
