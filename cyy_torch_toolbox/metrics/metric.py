import copy
from typing import Any

import torch

from ..hook import Hook


class Metric(Hook):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.__epoch_metrics: dict[int, dict[str, Any]] = {}
        self.__batch_metrics: dict[int, dict[str, Any]] = {}

    def get_epoch_metric(self, epoch: int, name: str, to_item: bool = True) -> Any:
        return self.get_metric(
            metric_type="epoch", key=epoch, name=name, to_item=to_item
        )

    def get_epoch_metrics(self, epoch: int, to_item: bool = True) -> dict[str, Any]:
        return self.get_metrics(metric_type="epoch", key=epoch, to_item=to_item)

    def get_metrics(self, metric_type: str, key: int, to_item: bool = True) -> dict[str, Any]:
        metric: dict[str, Any] = {}
        match metric_type:
            case "epoch":
                metric = copy.copy(self.__epoch_metrics.get(key, {}))
            case "batch":
                metric = copy.copy(self.__batch_metrics.get(key, {}))
            case _:
                raise RuntimeError(metric_type)
        if to_item:
            for k, v in metric.items():
                if isinstance(v, torch.Tensor):
                    metric[k] = v.item()
        for sub_hook in self._sub_hooks:
            sub_metric = sub_hook.get_metrics(
                metric_type=metric_type, key=key, to_item=to_item
            )
            for k, v in sub_metric.items():
                assert k not in metric
                metric[k] = v
        return metric

    def get_metric(
        self, metric_type: str, key: int, name: str, to_item: bool = True
    ) -> Any:
        metric: dict[str, Any] = {}
        match metric_type:
            case "epoch":
                metric = copy.copy(self.__epoch_metrics.get(key, {}))
            case "batch":
                metric = copy.copy(self.__batch_metrics.get(key, {}))
            case _:
                raise RuntimeError(metric_type)
        if name in metric:
            res = metric[name]
            if to_item and isinstance(res, torch.Tensor):
                return res.item()
            return res
        for sub_hook in self._sub_hooks:
            sub_metric = sub_hook.get_metrics(
                metric_type=metric_type, key=key, to_item=to_item
            )
            if name in sub_metric:
                res = sub_metric[name]
                if to_item and isinstance(res, torch.Tensor):
                    return res.item()
                return res
        return None

    def _set_epoch_metric(self, epoch: int, name: str, data: Any) -> None:
        if epoch not in self.__epoch_metrics:
            self.__epoch_metrics[epoch] = {}
        if isinstance(data, torch.Tensor):
            data = data.clone().detach()
        self.__epoch_metrics[epoch][name] = data

    def get_batch_metric(self, batch: int, name: str) -> Any:
        return self.get_metric(metric_type="batch", key=batch, name=name)

    def _set_batch_metric(self, batch_index: int, name: str, data: Any) -> None:
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
