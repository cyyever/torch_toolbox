import copy
import functools
from typing import Any

from cyy_naive_lib.log import get_logger

from cyy_torch_toolbox.executor import Executor
from cyy_torch_toolbox.ml_type import ExecutorHookPoint, StopExecutingException


class Inferencer(Executor):
    def inference(
        self, require_grad: bool = False, reduce_loss: bool = True, **kwargs: Any
    ) -> bool:
        succ_flag: bool = False
        with (
            torch.set_grad_enabled(require_grad),
            self.device_context,
            self.device_stream_context,
        ):
            try:
                self._prepare_execution(**kwargs)
                self._execute_epoch(
                    epoch=1,
                    require_grad=require_grad,
                    in_training=False,
                    reduce_loss=reduce_loss,
                )
                self.exec_hooks(hook_point=ExecutorHookPoint.AFTER_EXECUTE)
                succ_flag = True
            except StopExecutingException:
                get_logger().warning("stop inference")
            finally:
                self.wait_stream()
            return succ_flag

    def _get_backward_loss(self, result: dict) -> Any:
        return result["normalized_batch_loss"]

    def get_gradient(self) -> dict:
        old_hook_config = copy.copy(self.hook_config)
        succ: bool = self.inference(require_grad=True, use_performance_metric=False)
        self.hook_config = old_hook_config
        assert succ
        return self.model_util.get_gradient_dict()

    def get_sample_loss(self) -> dict:
        sample_loss: dict = {}
        old_hook_config = copy.copy(self.hook_config)
        name = "__cyy_collect_sample_loss"
        self.append_named_hook(
            hook_point=ExecutorHookPoint.AFTER_BATCH,
            name=name,
            fun=functools.partial(self._collect_sample_loss, sample_loss),
        )
        succ: bool = self.inference(
            require_grad=False, reduce_loss=False, use_performance_metric=False
        )
        self.remove_named_hook(name=name)
        self.hook_config = old_hook_config
        assert succ
        assert len(sample_loss) == self.dataset_size
        return sample_loss

    def _collect_sample_loss(
        self, sample_loss, result, sample_indices, **kwargs
    ) -> None:
        assert not result["is_averaged_loss"]
        sample_loss.update(zip(sample_indices, result["loss"]))
