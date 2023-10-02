import copy
from typing import Any

import torch
from cyy_torch_toolbox.hook import Hook


class Metric(Hook):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__epoch_metrics: dict = {}
        self.__batch_metrics: dict = {}

    def _get_output(self, result: dict) -> torch.Tensor:
        output = result["model_output"]
        logits = result.get("logits", None)
        if logits is not None:
            output = logits
        return output

    def get_epoch_metric(self, epoch: int, name: str | None = None) -> dict | Any:
        return self.get_metrics(metric_type="epoch", key=epoch, name=name)

    def get_metrics(
        self, metric_type: str, key: int, name: str | None = None
    ) -> dict | Any:
        metric: dict = {}
        match metric_type:
            case "epoch":
                metric = copy.copy(self.__epoch_metrics.get(key, {}))
            case "batch":
                metric = copy.copy(self.__batch_metrics.get(key, {}))
            case _:
                raise RuntimeError(metric_type)
        for k, v in metric.items():
            if isinstance(v, torch.Tensor):
                metric[k] = v.item()
        if name is not None and name in metric:
            return metric.get(name, None)
        for sub_hook in self._sub_hooks:
            sub_metric = sub_hook.get_metrics(
                metric_type=metric_type, key=key, name=None
            )
            if sub_metric:
                for k, v in sub_metric.items():
                    assert k not in metric
                    metric[k] = v
        if name is not None:
            return metric.get(name, None)
        return metric

    def _set_epoch_metric(self, epoch, name, data) -> None:
        if epoch not in self.__epoch_metrics:
            self.__epoch_metrics[epoch] = {}
        if isinstance(data, torch.Tensor):
            data = data.clone().detach()
        self.__epoch_metrics[epoch][name] = data

    def get_batch_metric(self, batch: int, name: str) -> Any:
        return self.get_metrics(metric_type="batch", key=batch, name=name)

    def _set_batch_metric(self, batch_index, name, data) -> None:
        if batch_index not in self.__batch_metrics:
            self.__batch_metrics[batch_index] = {}
        self.__batch_metrics[batch_index][name] = data

    def _before_execute(self, **__) -> None:
        self.clear_metric()

    def clear_metric(
        self, metric_type: str | None = None, key: int | None = None
    ) -> None:
        for sub_hook in self._sub_hooks:
            if hasattr(sub_hook, "clear_metric"):
                sub_hook.clear_metric(metric_type=metric_type, key=key)
        match metric_type:
            case "epoch":
                self.__epoch_metrics.pop(key, None)
            case "batch":
                self.__batch_metrics.pop(key, None)
            case None:
                self.__batch_metrics.clear()
                self.__epoch_metrics.clear()
            case _:
                raise RuntimeError(metric_type)
