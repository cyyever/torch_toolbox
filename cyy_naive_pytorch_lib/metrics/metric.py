from typing import Callable

from ml_type import ModelExecutorCallbackPoint
from model_executor import ModelExecutor


class Metric:
    def __init__(self, model_exetutor: ModelExecutor):
        self._model_executor = model_exetutor
        self.__callback_names: list = []

    def add_callback(self, cb_point: ModelExecutorCallbackPoint, cb: Callable):
        name = self.__class__.__name__ + "." + str(cb)
        self._model_executor.add_named_callback(cb_point, name, cb)
        self.__callback_names.append(name)

    def remove_callbacks(self):
        for name in self.__callback_names:
            self._model_executor.remove_callback(name)
