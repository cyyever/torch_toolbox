from typing import Any

from cyy_torch_toolbox.hook import Hook


class Metric(Hook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__epoch_metrics: dict = {}
        self.__batch_metrics: dict = {}

    def get_epoch_metric(self, epoch: int, name: str | None = None) -> Any:
        epoch_data = self.__epoch_metrics.get(epoch, None)
        if epoch_data is None:
            return None
        if name is None:
            return epoch_data
        return epoch_data.get(name, None)

    def get_epoch_metrics(self, epoch: int) -> dict | None:
        epoch_metrics = self.get_epoch_metric(epoch=epoch)
        if epoch_metrics is None:
            epoch_metrics = {}
        for sub_hook in self._sub_hooks:
            if hasattr(sub_hook, "get_epoch_metric"):
                epoch_metric = sub_hook.get_epoch_metric(epoch=epoch)
                if epoch_metric:
                    epoch_metrics |= epoch_metric
        if epoch_metrics:
            return epoch_metrics
        return None

    def _set_epoch_metric(self, epoch, name, data):
        if epoch not in self.__epoch_metrics:
            self.__epoch_metrics[epoch] = {}
        self.__epoch_metrics[epoch][name] = data

    def get_batch_metric(self, batch: int, name: str | None = None) -> Any:
        batch_data = self.__batch_metrics.get(batch, None)
        if batch_data is None:
            return None
        if name is None:
            return batch_data
        return batch_data.get(name, None)

    def get_batch_metrics(self, batch: int) -> dict | None:
        batch_metrics = self.get_batch_metric(batch=batch)
        if batch_metrics is None:
            batch_metrics = {}
        for sub_hook in self._sub_hooks:
            if hasattr(sub_hook, "get_batch_metric"):
                batch_metric = sub_hook.get_batch_metric(batch=batch)
                if batch_metric:
                    batch_metrics |= batch_metric
        if batch_metrics:
            return batch_metrics
        return None

    def _set_batch_metric(self, batch, name, data):
        if batch not in self.__batch_metrics:
            self.__batch_metrics[batch] = {}
        self.__batch_metrics[batch][name] = data

    def _before_execute(self, **__):
        self.clear_metric()

    def clear_metric(self):
        for sub_hook in self._sub_hooks:
            if hasattr(sub_hook, "clear_metric"):
                sub_hook.clear_metric()
        self.__epoch_metrics.clear()
        self.__batch_metrics.clear()
