from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(kw_only=True)
class Transform:
    fun: Callable
    name: str = ""
    cacheable: bool = False
    component: str | None = None

    def __str__(self) -> str:
        fun_name = self.name if self.name else str(self.fun)
        return f"name:{fun_name} cacheable:{self.cacheable}"

    def __call__(self, data: Any) -> Any:
        assert data is not None
        if self.component is not None:
            data[self.component] = self.fun(data[self.component])
            return data
        return self.fun(data)


class SampleTransform(Transform):
    pass


class DatasetTransform(Transform):
    pass


class BatchTransform(Transform):
    pass
