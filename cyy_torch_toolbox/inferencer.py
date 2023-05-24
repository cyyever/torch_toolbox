import contextlib
from typing import Any

import torch
from cyy_naive_lib.log import get_logger

from cyy_torch_toolbox.executor import Executor
from cyy_torch_toolbox.ml_type import ExecutorHookPoint, StopExecutingException


class Inferencer(Executor):
    def inference(self, use_grad: bool = False, **kwargs: Any) -> bool:
        has_failure: bool = False
        try:
            self._prepare_execution(**kwargs)
            with (
                torch.set_grad_enabled(use_grad),
                torch.cuda.device(self.device)
                if self.cuda_stream is not None
                else contextlib.nullcontext(),
                torch.cuda.stream(self.cuda_stream),
            ):
                self.model.zero_grad(set_to_none=True)
                self._execute_epoch(epoch=1, need_backward=use_grad, in_training=False)
            self.exec_hooks(ExecutorHookPoint.AFTER_EXECUTE)
        except StopExecutingException:
            get_logger().warning("stop inference")
            has_failure = True
        finally:
            self.wait_stream()
        return not has_failure

    def get_optimizer(self) -> Any:
        return None

    def get_lr_scheduler(self) -> Any:
        return None

    def _get_backward_loss(self, result):
        return result["normalized_batch_loss"]

    def get_gradient(self) -> dict:
        normal_stop: bool = self.inference(use_grad=True)
        assert normal_stop
        return self.model_util.get_gradient_dict()
