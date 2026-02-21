from typing import override

import torch
from torch.optim import SGD

from .optimizer import MetaOptimizer


class MetaSGD(MetaOptimizer):
    def __init__(self, optimizer: SGD) -> None:
        self.__optimizer = optimizer

    @override
    def step(self) -> list[list[torch.Tensor]]:
        results: list[list[torch.Tensor]] = []
        for param_group in self.__optimizer.param_groups:
            nesterov = param_group["nesterov"]
            maximize = param_group["maximize"]
            params = param_group["params"]
            weight_decay = param_group["weight_decay"]
            momentum = param_group["momentum"]
            lr = param_group["lr"]
            dampening = param_group["dampening"]
            new_params = []
            for param in params:
                assert param.grad is not None
                d_p = param.grad
                d_p = d_p if not maximize else -d_p
                if weight_decay != 0:
                    d_p = d_p.add(param, alpha=weight_decay)

                if momentum != 0:
                    state = self.__optimizer.state[param]
                    buf = state.get("momentum_buffer", None)
                    if buf is not None:
                        buf = buf.mul(momentum).add(d_p, alpha=1 - dampening)
                    else:
                        buf = d_p
                    d_p = d_p.add(buf, alpha=momentum) if nesterov else buf
                new_params.append(param.add(d_p, alpha=-lr))
            results.append(new_params)
        return results
