import torch


class MetaOptimizer:
    def step(self) -> list[list[torch.Tensor]]:
        raise NotImplementedError()
