from difflib import get_close_matches
from typing import Any


class Factory:
    def __init__(self) -> None:
        self.data: dict = {}

    def register(self, key, value) -> None:
        match value:
            case list():
                if key not in self.data:
                    self.data[key] = value
                else:
                    assert isinstance(self.data[key], list)
                    self.data[key] += value
            case dict():
                assert isinstance(self.data[key], dict)
                self.data[key].update(value)
            case _:
                self.data[key] = value

    def get(self, key: Any, case_sensitive: bool = True) -> Any:
        if not case_sensitive:
            key = self._lower_key(key)
        return self.data.get(key, None)

    def get_similar_keys(self, key: str) -> list[str]:
        return get_close_matches(key, self.data.keys())

    @classmethod
    def _lower_key(cls, key: Any) -> Any:
        match key:
            case str():
                return key.lower()
            case tuple():
                return tuple(map(cls._lower_key, key))
            case _:
                raise NotImplementedError()
