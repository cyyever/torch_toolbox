import torch
from cyy_naive_lib.reflection import get_class_attrs

from ..ml_type.factory import Factory


class OptimizerFactory(Factory):
    def __init__(self) -> None:
        super().__init__()
        for name, cls in get_class_attrs(
            torch.optim,
            filter_fun=lambda _, v: issubclass(v, torch.optim.Optimizer),
        ).items():
            self.register(name, cls)

    def register(self, key: str, value: type[torch.optim.Optimizer]) -> None:
        assert key not in self.data, key
        super().register(key, value)


global_optimizer_factory = OptimizerFactory()


def get_optimizer_names() -> list[str]:
    names = global_optimizer_factory.get_keys()
    assert all(isinstance(name, str) for name in names)
    return names
