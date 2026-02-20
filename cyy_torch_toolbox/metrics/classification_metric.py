from typing import TYPE_CHECKING, Any, Literal

import torch

if TYPE_CHECKING:
    from ..executor import Executor
import torchmetrics.metric

from ..ml_type import ModelType
from .metric import Metric


class ClassificationMetric(Metric):
    _metric: None | torchmetrics.metric.Metric = None

    _task_type: Literal["binary", "multiclass", "multilabel"] | None = None

    @property
    def metric(self) -> torchmetrics.metric.Metric:
        assert self._metric is not None
        return self._metric

    def _before_execution(self, **kwargs: Any) -> None:
        executor = kwargs["executor"]
        if executor.running_model_evaluator.model_type != ModelType.Classification:
            self.disable()

    def _before_epoch(self, **kwargs: Any) -> None:
        self._metric = None

    @torch.no_grad()
    def _get_task(self, executor: "Executor") -> Literal["binary", "multiclass", "multilabel"]:
        if self._task_type is not None:
            return self._task_type
        if (
            executor.running_model_evaluator.model_type != ModelType.TokenClassification
            and executor.dataset_collection.is_multilabel()
        ):
            self._task_type = "multilabel"
            return self._task_type
        if executor.dataset_collection.label_number <= 2:
            self._task_type = "binary"
            return self._task_type
        self._task_type = "multiclass"
        return self._task_type

    @torch.no_grad()
    def _get_output(self, executor: "Executor", result: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        targets: torch.Tensor = result["targets"].detach()
        if targets.dtype is torch.float:
            targets = targets.to(dtype=torch.long, non_blocking=True)

        output = result.get("logits")
        if output is None:
            output = result.get("original_output")
        assert isinstance(output, torch.Tensor)
        output = output.detach()
        if self._task_type == "multiclass":
            output = output.view(-1, output.shape[-1])
            targets = targets.view(-1)

        if len(output.shape) == 2 and output.shape[-1] == 1:
            output = output.view(-1)
        assert isinstance(targets, torch.Tensor)
        if len(targets.shape) <= 1:
            output = output.to("cpu", non_blocking=True)
            targets = targets.to("cpu", non_blocking=True)
            mask = targets != -100
            output = output[mask]
            targets = targets[mask]

        with executor.device:
            if executor.dataset_collection.label_number <= 2 and output.shape[-1] == 2:
                output = torch.argmax(output, dim=-1)
        return output, targets

    def _get_metric_kwargs(self, executor: "Executor") -> dict[str, Any]:
        task = self._get_task(executor)
        kwargs = {"task": task}
        if task == "multilabel":
            kwargs["num_labels"] = executor.dataset_collection.label_number
        else:
            kwargs["num_classes"] = executor.dataset_collection.label_number
        return kwargs
