import contextlib
from typing import Any, Self

import torch
from cyy_naive_lib import Decorator
from cyy_naive_lib.log import log_debug

from .evaluator import ModelEvaluator


class TF32Context:
    def __init__(self) -> None:
        self.__old_cudnn_allow_tf32 = False
        self.__old_cuda_allow_tf32 = False

    def __enter__(self) -> Self:
        self.__old_cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
        self.__old_cuda_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        torch.backends.cudnn.allow_tf32 = self.__old_cudnn_allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.__old_cuda_allow_tf32


class AMPModelEvaluator(Decorator):
    def __init__(self, evaluator: ModelEvaluator) -> None:
        super().__init__(evaluator)
        self.__amp_ctx: None | torch.autocast = None
        self.__scaler: None | torch.GradScaler = None
        self.use_tf32: bool = True
        self.check_inf: bool = False

    @property
    def evaluator(self) -> ModelEvaluator:
        return self._decorator_object

    def __call__(self, /, device: torch.device, **kwargs: Any) -> Any:
        if self.__amp_ctx is None or device.type != self.__amp_ctx.device:
            self.__amp_ctx = torch.autocast(device_type=device.type)
        with (
            TF32Context()
            if self.use_tf32 and "cuda" in device.type.lower()
            else contextlib.nullcontext(),
            self.__amp_ctx,
        ):
            return self.evaluator.__call__(device=device, **kwargs)

    def backward_and_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        **backward_kwargs: Any,
    ) -> None:
        assert self.__amp_ctx is not None
        if self.__scaler is None:
            self.__scaler = torch.GradScaler(device=self.__amp_ctx.device)
        while True:
            optimizer.zero_grad(set_to_none=True)
            self.evaluator.backward(
                loss=self.__scaler.scale(loss), retain_graph=True, **backward_kwargs
            )
            self.__scaler.step(optimizer)
            has_inf = 0
            if self.check_inf:
                has_inf = sum(
                    found_inf.item()
                    for state in self.__scaler._per_optimizer_states.values()
                    for found_inf in state["found_inf_per_device"].values()
                )

            # Updates the scale for next iteration.
            self.__scaler.update()
            if has_inf > 0:
                log_debug("found inf in AMP, scale is %s", self.__scaler._scale)
                continue
            break
