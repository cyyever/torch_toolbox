from cyy_torch_toolbox.hook import Hook


class Metric(Hook):
    def __init__(self):
        super().__init__()
        self.__epoch_metrics: dict = dict()
        self._is_cyy_torch_toolbox_metric = True

    def _before_execute(self, **__):
        self.__epoch_metrics.clear()

    def get_epoch_metric(self, epoch: int, name=None):
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

    def clear_metric(self):
        for name in dir(self):
            attr = getattr(self, name)
            if hasattr(attr, "_is_cyy_torch_toolbox_metric"):
                attr.clear_metric()
        self.__epoch_metrics.clear()
