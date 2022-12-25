from typing import Any

from cyy_torch_toolbox.hook import HookCollection


class ModelExecutorBase(HookCollection):
    def __init__(self):
        super().__init__()
        self.__data: dict = {}

    def get_data(self, key: str, default_value: Any = None) -> Any:
        return self.__data.get(key, default_value)

    def pop_data(self, key: str, default_value: Any = None) -> Any:
        return self.__data.pop(key, default_value)

    def set_data(self, key: str, value: Any) -> None:
        self.__data[key] = value

    def remove_data(self, key: str) -> None:
        self.__data.pop(key, None)

    def has_data(self, key: str) -> bool:
        return key in self.__data

    def clear_data(self):
        self.__data.clear()
