import torch
from cyy_naive_lib.log import get_logger

from cyy_torch_toolbox.ml_type import (ModelExecutorHookPoint,
                                       StopExecutingException)
from cyy_torch_toolbox.model_executor import ModelExecutor


class Inferencer(ModelExecutor):
    # def __init__(self, *args, batch_size=8, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.__batch_size = batch_size

    # def _get_batch_size(self) -> int:
    #     return self.__batch_size

    def inference(self, use_grad: bool = False, epoch: int = 1, **kwargs: dict) -> bool:
        error_return: bool = False
        try:
            self._prepare_execution(**kwargs)
            with (
                torch.set_grad_enabled(use_grad),
                torch.cuda.stream(self.cuda_stream),
            ):
                self.model.zero_grad(set_to_none=True)
                self._execute_epoch(epoch=epoch, need_backward=use_grad)
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

    def get_optimizer(self):
        return None

    def get_lr_scheduler(self):
        return None
