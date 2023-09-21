import copy
from typing import Any

import torch
from cyy_torch_toolbox.hook import Hook


class Metric(Hook):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__epoch_metrics: dict = {}
        self.__batch_metrics: dict = {}

    def get_epoch_metric(self, epoch: int, name: str | None = None) -> dict | Any:
        epoch_metric = copy.deepcopy(self.__epoch_metrics.get(epoch, {}))
        for sub_hook in self._sub_hooks:
            if hasattr(sub_hook, "get_epoch_metric"):
                sub_epoch_metric = sub_hook.get_epoch_metric(epoch=epoch)
                for k, v in sub_epoch_metric.items():
                    assert k not in epoch_metric
                    epoch_metric[k] = v
        for k, v in epoch_metric.items():
            if isinstance(v, torch.Tensor):
                epoch_metric[k] = v.item()
        if name is None:
            return epoch_metric
        return epoch_metric.get(name, None)

    def _set_epoch_metric(self, epoch, name, data) -> None:
        if epoch not in self.__epoch_metrics:
            self.__epoch_metrics[epoch] = {}
        if isinstance(data, torch.Tensor):
            data = data.clone().detach()
        self.__epoch_metrics[epoch][name] = data

    def get_batch_metric(self, batch: int, name: str) -> Any:
        return self.get_batch_metrics(batch=batch).get(name, None)

    def get_batch_metrics(self, batch: int) -> dict:
        batch_metrics = self.__batch_metrics.get(batch, {})
        for sub_hook in self._sub_hooks:
            if hasattr(sub_hook, "get_batch_metrics"):
                batch_metrics |= sub_hook.get_batch_metrics(batch=batch)
        return batch_metrics

    def _set_batch_metric(self, batch, name, data) -> None:
        if batch not in self.__batch_metrics:
            self.__batch_metrics[batch] = {}
        self.__batch_metrics[batch][name] = data

    def _before_execute(self, **__) -> None:
        self.clear_metric()

    def clear_metric(self) -> None:
        for sub_hook in self._sub_hooks:
            if hasattr(sub_hook, "clear_metric"):
                sub_hook.clear_metric()
        self.__epoch_metrics.clear()
        self.__batch_metrics.clear()
