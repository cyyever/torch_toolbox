from torch.optim import SGD, Optimizer

from .optimizer import MetaOptimizer
from .sgd import MetaSGD


def get_meta_optimizer(optimizer: Optimizer) -> MetaOptimizer:
    match optimizer:
        case SGD():
            return MetaSGD(optimizer)
        case _:
            raise RuntimeError()
