from typing import Any

import torch

from ..ml_type import ModelType
from .metric import Metric


class AccuracyMetric(Metric):
    __correct_count: int | torch.Tensor | None = None
    __dataset_size: int | torch.Tensor = 0

    def _before_epoch(self, **kwargs: Any) -> None:
        self.__dataset_size = 0
        self.__correct_count = None

    @torch.inference_mode()
    def _after_batch(self, result: dict[str, Any], **kwargs: Any) -> None:
        targets = result["targets"]
        output = result.get("model_output")
        if output is None:
            output = result.get("logits")
        assert isinstance(output, torch.Tensor)
        assert isinstance(targets, torch.Tensor)
        correct_count: int | torch.Tensor = 0
        executor = kwargs["executor"]
        if executor.running_model_evaluator.model_type == ModelType.TokenClassification:
            max_output = torch.argmax(output, dim=2)
            mask = targets != -100
            correct_count = torch.eq(max_output[mask], targets[mask]).view(-1).sum()
            self.__dataset_size += mask.sum()
        else:
            if output.shape == targets.shape:
                if len(targets.shape) == 2:
                    for idx, maxidx in enumerate(torch.argmax(output, dim=1)):
                        if targets[idx][maxidx] == 1:
                            correct_count += 1
                elif len(targets.shape) <= 1:
                    correct_count = (
                        torch.eq(torch.round(output.sigmoid()), targets).view(-1).sum()
                    )
                else:
                    raise NotImplementedError()
            else:
                assert len(output.shape) == 2
                correct_count = (
                    torch.eq(torch.max(output, dim=1)[1], targets.view(-1))
                    .view(-1)
                    .sum()
                )
            self.__dataset_size += targets.shape[0]
        if self.__correct_count is None:
            self.__correct_count = correct_count
        else:
            self.__correct_count += correct_count

    def _after_epoch(self, **kwargs: Any) -> None:
        epoch = kwargs["epoch"]
        assert self.__correct_count is not None
        accuracy = self.__correct_count / self.__dataset_size
        self._set_epoch_metric(epoch, "accuracy", accuracy)
