from typing import Any

import torch
import torch.amp
from cyy_naive_lib.log import log_warning

from .evaluator import ModelEvaluator


class AMPModelEvaluator:
    def __init__(self, evaluator: ModelEvaluator) -> None:
        self.evaluator: ModelEvaluator = evaluator
        self.__amp_ctx: None | torch.autocast = None
        self.__scaler: None | torch.GradScaler = None

    def __getattr__(self, name):
        if name == "evaluator":
            raise AttributeError()
        return getattr(self.evaluator, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        device: torch.device = kwargs["device"]
        if self.__amp_ctx is None or device.type != self.__amp_ctx.device:
            self.__amp_ctx = torch.autocast(device_type=device.type)
        assert self.__amp_ctx is not None
        with self.__amp_ctx:
            return self.evaluator.__call__(*args, **kwargs)

    def backward_and_step(
        self,
        loss,
        optimizer: torch.optim.Optimizer,
        **backward_kwargs,
    ) -> Any:
        assert self.__amp_ctx is not None
        if self.__scaler is None and self.__amp_ctx is not None:
            self.__scaler = torch.GradScaler(device=self.__amp_ctx.device)
        if self.__scaler is None:
            return self.evaluator.backward_and_step(
                loss=loss, optimizer=optimizer, **backward_kwargs
            )
        while True:
            optimizer.zero_grad(set_to_none=True)
            self.evaluator.backward(
                loss=self.__scaler.scale(loss), retain_graph=True, **backward_kwargs
            )
            self.__scaler.step(optimizer)
            has_inf = sum(
                found_inf.item()
                for state in self.__scaler._per_optimizer_states.values()
                for found_inf in state["found_inf_per_device"].values()
            )

            # Updates the scale for next iteration.
            self.__scaler.update()
            if has_inf > 0:
                log_warning("found inf in AMP, scale is %s", self.__scaler._scale)
                continue
            break
        return None
