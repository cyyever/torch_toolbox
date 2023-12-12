from torch.optim import SGD, Adam, Optimizer

from .adam import MetaAdam
from .optimizer import MetaOptimizer
from .sgd import MetaSGD


def get_meta_optimizer(optimizer: Optimizer) -> MetaOptimizer:
    match optimizer:
        case SGD():
            return MetaSGD(optimizer)
        case Adam():
            return MetaAdam(optimizer)
        case _:
            raise RuntimeError()
