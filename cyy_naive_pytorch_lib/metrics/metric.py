from callback import Callback


class Metric2(Callback):
    def __init__(self):
        super().__init__()
        self.__epoch_metrics: dict = dict()

    def clear(self):
        self.__epoch_metrics.clear()

    def _before_execute(self, *args, **kwargs):
        self.clear()

    def get_epoch_metric(self, epoch, name):
        epoch_data = self.__epoch_metrics.get(epoch, None)
        if epoch_data is None:
            return None
        return epoch_data.get(name, None)

    def _set_epoch_metric(self, epoch, name, data):
        if epoch not in self.__epoch_metrics:
            self.__epoch_metrics[epoch] = dict()
        self.__epoch_metrics[epoch][name] = data
