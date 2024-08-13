import copy
from typing import Any, Self


class ConfigBase:
    def __init__(self) -> None:
        self.__old_config: Any = None

    def __enter__(self) -> Self:
        self.__old_config = copy.deepcopy(self)
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        for name in dir(self):
            if not name.startswith("_"):
                setattr(self, name, getattr(self.__old_config, name))
        self.__old_config = None
