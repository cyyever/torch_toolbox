import torch
from cyy_naive_lib.log import get_logger

from cyy_torch_toolbox.ml_type import (ModelExecutorHookPoint,
                                       StopExecutingException)
from cyy_torch_toolbox.model_executor import ModelExecutor


class Inferencer(ModelExecutor):
    def inference(self, use_grad=False, epoch=1, **kwargs) -> bool:
        self._prepare_execution(**kwargs)
        self.exec_hooks(ModelExecutorHookPoint.BEFORE_EXECUTE)
        early_stop: bool = False
        with (torch.set_grad_enabled(use_grad), torch.cuda.stream(self.cuda_stream)):
            self.model.zero_grad(set_to_none=True)
            try:
                self._execute_epoch(
                    epoch=epoch, need_backward=use_grad, in_training=False
                )
            except StopExecutingException:
                get_logger().warning("stop inference")
                early_stop = True
            finally:
                self._wait_stream()
            self.exec_hooks(ModelExecutorHookPoint.AFTER_EXECUTE)
        return not early_stop

    def _get_backward_loss(self, result, need_backward):
        if need_backward:
            return result["normalized_batch_loss"]
        return None

    def get_gradient(self):
        normal_stop: bool = self.inference(use_grad=True)
        assert normal_stop
        return self.model_util.get_gradient_list()

    def get_optimizer(self):
        return None

    def get_lr_scheduler(self):
        return None
