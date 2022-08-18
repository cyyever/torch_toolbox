import torch
from cyy_naive_lib.log import get_logger

from cyy_torch_toolbox.ml_type import (ModelExecutorHookPoint,
                                       StopExecutingException)
from cyy_torch_toolbox.model_executor import ModelExecutor


class Inferencer(ModelExecutor):
    _use_grad = False

    def inference(self, use_grad=False, epoch=None, **kwargs):
        self._use_grad = use_grad
        self._prepare_execution(**kwargs)
        if epoch is None:
            epoch = 1
        self.exec_hooks(ModelExecutorHookPoint.BEFORE_EXECUTE)
        with (torch.set_grad_enabled(use_grad), torch.cuda.stream(self.cuda_stream)):
            self.model.zero_grad(set_to_none=True)
            try:
                self._execute_epoch(epoch=epoch, need_backward=self._use_grad)
            except StopExecutingException:
                get_logger().warning("stop inference")
            finally:
                self._wait_stream()
            self.exec_hooks(ModelExecutorHookPoint.AFTER_EXECUTE)

    def _get_backward_loss(self, result):
        if self._use_grad:
            return result["normalized_batch_loss"]
        return None

    def get_gradient(self):
        self.inference(use_grad=True)
        return self.model_util.get_gradient_list()

    def get_optimizer(self):
        return None

    def get_lr_scheduler(self):
        return None
