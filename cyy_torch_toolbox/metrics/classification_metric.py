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

    @torch.no_grad()
    def _get_task(self, executor) -> Literal["binary", "multiclass", "multilabel"]:
        if (
            executor.running_model_evaluator.model_type != ModelType.TokenClassification
            and executor.dataset_collection.is_mutilabel()
        ):
            return "multilabel"
        if executor.dataset_collection.label_number <= 2:
            return "binary"
        return "multiclass"

    @torch.no_grad()
    def _get_output(self, executor, result: dict) -> tuple[torch.Tensor, torch.Tensor]:
        targets: torch.Tensor = result["targets"].detach()
        if targets.dtype is torch.float:
            targets = targets.to(dtype=torch.long, non_blocking=True)

        output = result.get("logits")
        if output is None:
            output = result.get("original_output")
        assert isinstance(output, torch.Tensor)
        output = output.detach()
        if len(output.shape) == 2 and output.shape[-1] == 1:
            output = output.view(-1)
        assert isinstance(targets, torch.Tensor)
        if len(targets.shape) <= 1:
            mask = targets != -100
            output = output[mask]
            targets = targets[mask]

        with executor.device:
            if executor.dataset_collection.label_number <= 2 and output.shape[-1] == 2:
                output = torch.argmax(output, dim=-1)
        return output, targets

    def _get_metric_kwargs(self, executor) -> dict:
        task = self._get_task(executor)
        kwargs = {"task": task}
        if task == "multilabel":
            kwargs["num_labels"] = executor.dataset_collection.label_number
        else:
            kwargs["num_classes"] = executor.dataset_collection.label_number
        return kwargs
