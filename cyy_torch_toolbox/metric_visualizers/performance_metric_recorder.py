import json
import os
import re

import torch
from cyy_torch_toolbox.ml_type import MachineLearningPhase

from .metric_visualizer import MetricVisualizer


class PerformanceMetricRecorder(MetricVisualizer):
    def _after_epoch(self, model_executor, epoch, **kwargs):
        phase_str = "training"
        if model_executor.phase == MachineLearningPhase.Validation:
            phase_str = "validation"
        elif model_executor.phase == MachineLearningPhase.Test:
            phase_str = "test"
        prefix = re.sub(r"[: ,]+$", "", self._prefix)
        prefix = re.sub(r"[: ,]+", "_", prefix)

        json_filename = os.path.join(
            self._data_dir, prefix, phase_str, "performance_metric.json"
        )
        os.makedirs(os.path.dirname(json_filename), exist_ok=True)
        json_record = {}
        if os.path.isfile(json_filename):
            with open(json_filename, "rt", encoding="utf8") as f:
                json_record = json.load(f)
        epoch_metrics = model_executor.performance_metric.get_epoch_metrics(epoch)
        if not epoch_metrics:
            return
        for k, value in epoch_metrics.items():
            if k not in json_record:
                json_record[k] = {}
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.item()
            json_record[k][epoch] = value
        with open(json_filename, "wt", encoding="utf8") as f:
            json.dump(json_record, f)
