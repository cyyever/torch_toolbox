from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(kw_only=True)
class Transform:
    fun: Callable
    name: str = ""
    cacheable: bool = False

    def __str__(self) -> str:
        fun_name = self.name if self.name else str(self.fun)
        return f"name:{fun_name} cacheable:{self.cacheable}"

    def __call__(self, data: Any) -> Any:
        return self.fun(data)


@dataclass(kw_only=True)
class SampleTransform(Transform):
    component: str | None = None

    def __call__(self, data: Any) -> Any:
        if self.component is not None:
            data[self.component] = self.fun(data[self.component])
            return data
        return super().__call__(data)


class DatasetTransform(Transform):
    pass


class BatchTransform(Transform):
    pass
