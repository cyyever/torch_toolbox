import torch
from torch.optim import Adam

from .optimizer import MetaOptimizer


class MetaAdam(MetaOptimizer):
    def __init__(self, optimizer: Adam) -> None:
        self.__optimizer = optimizer

    def step(self) -> list[list]:
        results: list[list] = []
        for group in self.__optimizer.param_groups:
            params = group["params"]
            weight_decay = group["weight_decay"]
            maximize = group["maximize"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            new_params = []
            for param in params:
                assert not torch.is_complex(param)
                assert param.grad is not None

                state = self.__optimizer.state[param]

                # Exponential moving average of gradient values
                exp_avg = state.get(
                    "exp_avg",
                    torch.zeros_like(param, memory_format=torch.preserve_format),
                )
                # Exponential moving average of squared gradient values
                exp_avg_sq = state.get(
                    "exp_avg_sq",
                    torch.zeros_like(param, memory_format=torch.preserve_format),
                )
                step = state.get("step", torch.tensor(0.0, dtype=torch.float32))

                # update step
                step = step + 1

                grad = param.grad if not maximize else -param.grad

                if weight_decay != 0:
                    grad = grad.add(param, alpha=weight_decay)

                # Decay the first and second moment running average coefficient
                exp_avg = exp_avg.lerp(grad, 1 - beta1)
                exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(
                    grad, grad.conj(), value=1 - beta2
                )

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                step_size = lr / bias_correction1

                bias_correction2_sqrt = bias_correction2.sqrt()

                if group["amsgrad"]:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    max_exp_avg_sq = state.get(
                        "max_exp_avg_sq",
                        torch.zeros_like(param, memory_format=torch.preserve_format),
                    )
                    # Maintains the maximum of all 2nd moment running avg. till now
                    max_exp_avg_sq = torch.maximum(max_exp_avg_sq, exp_avg_sq)

                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / bias_correction2_sqrt).add(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add(eps)

                param = param.addcdiv(exp_avg, denom, value=-step_size)
                new_params.append(param)
            results.append(new_params)
        return results
