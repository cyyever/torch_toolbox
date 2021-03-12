from typing import Callable

from ml_type import ModelExecutorCallbackPoint
from model_executor import ModelExecutor


class Metric:
    def __init__(self, model_exetutor: ModelExecutor):
        self._model_executor = model_exetutor
        self.__callback_names: list = []
        self.__epoch_metrics: dict = dict()
        self.add_callback(
            ModelExecutorCallbackPoint.BEFORE_EXECUTE,
            lambda *args, **kwargs: self.clear(),
        )

    def add_callback(self, cb_point: ModelExecutorCallbackPoint, cb: Callable):
        name = self.__class__.__name__ + "." + str(cb)
        self._model_executor.add_named_callback(cb_point, name, cb)
        self.__callback_names.append(name)

    def remove_callbacks(self):
        for name in self.__callback_names:
            self._model_executor.remove_callback(name)

    def clear(self):
        self.__epoch_metrics.clear()

    def get_epoch_metric(self, epoch, name):
        epoch_data = self.__epoch_metrics.get(epoch, None)
        if epoch_data is None:
            return None
        return epoch_data.get(name, None)

    def _set_epoch_metric(self, epoch, name, data):
        if epoch not in self.__epoch_metrics:
            self.__epoch_metrics[epoch] = dict()
        self.__epoch_metrics[epoch][name] = data
