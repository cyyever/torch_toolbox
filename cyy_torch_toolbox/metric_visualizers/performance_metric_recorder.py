import json
import os
import re
from typing import Any

import torch

from ..ml_type import MachineLearningPhase
from .metric_visualizer import MetricVisualizer


class PerformanceMetricRecorder(MetricVisualizer):
    def _after_epoch(self, executor, epoch, **kwargs: Any) -> None:
        prefix = re.sub(r"[: ,]+$", "", self._prefix)
        prefix = re.sub(r"[: ,]+", "_", prefix)

        assert self._data_dir is not None
        json_filename = os.path.join(
            self._data_dir, prefix, str(executor.phase), "performance_metric.json"
        )
        os.makedirs(os.path.dirname(json_filename), exist_ok=True)
        json_record = {}
        if os.path.isfile(json_filename):
            with open(json_filename, encoding="utf8") as f:
                json_record = json.load(f)
        epoch_metrics = executor.performance_metric.get_epoch_metrics(epoch)
        if not epoch_metrics and executor.phase != MachineLearningPhase.Training:
            epoch_metrics = executor.performance_metric.get_epoch_metrics(epoch=1)
        if not epoch_metrics:
            return
        for k, value in epoch_metrics.items():
            if k not in json_record:
                json_record[k] = {}
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().item()
            json_record[k][epoch] = value
        with open(json_filename, "w", encoding="utf8") as f:
            json.dump(json_record, f)
