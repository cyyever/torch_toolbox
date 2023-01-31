import contextlib

import torch
from cyy_naive_lib.log import get_logger

from cyy_torch_toolbox.ml_type import (ModelExecutorHookPoint,
                                       StopExecutingException)
from cyy_torch_toolbox.model_executor import ModelExecutor


class Inferencer(ModelExecutor):
    def inference(self, use_grad: bool = False, **kwargs: dict) -> bool:
        error_return: bool = False
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
            self.exec_hooks(ModelExecutorHookPoint.AFTER_EXECUTE)
        except StopExecutingException:
            get_logger().warning("stop inference")
            error_return = True
        finally:
            self._wait_stream()
        return not error_return

    def _get_backward_loss(self, result):
        return result["normalized_batch_loss"]

    def get_gradient(self):
        normal_stop: bool = self.inference(use_grad=True)
        assert normal_stop
        return self.model_util.get_gradient_list()
