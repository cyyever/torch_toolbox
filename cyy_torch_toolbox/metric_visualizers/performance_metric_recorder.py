import re
from typing import Any

import torch
from cyy_naive_lib.storage import load_json, save_json

from ..ml_type import MachineLearningPhase
from .metric_visualizer import MetricVisualizer


class PerformanceMetricRecorder(MetricVisualizer):
    def _after_epoch(self, executor: Any, epoch: int, **kwargs: Any) -> None:
        prefix = re.sub(r"[: ,]+$", "", self._prefix)
        prefix = re.sub(r"[: ,]+", "_", prefix)

        assert self._data_dir is not None
        json_path = (
            self._data_dir / prefix / str(executor.phase) / "performance_metric.json"
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_record = load_json(json_path) if json_path.is_file() else {}
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
        save_json(json_record, json_path, backup=False)
