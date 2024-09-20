import functools

import torch
from cyy_naive_lib.log import log_warning

from .executor import Executor
from .ml_type import (EvaluationMode, ExecutorHookPoint, ModelGradient,
                      StopExecutingException)


class Inferencer(Executor):
    def inference(
        self,
        evaluation_mode: EvaluationMode = EvaluationMode.Test,
    ) -> bool:
        succ_flag: bool = False
        require_grad: bool = EvaluationMode != EvaluationMode.Test
        with (
            torch.set_grad_enabled(require_grad),
            self.device_context,
            self.stream_context,
        ):
            try:
                self._prepare_execution()
                self._execute_epoch(epoch=1, evaluation_mode=evaluation_mode)
                self.exec_hooks(hook_point=ExecutorHookPoint.AFTER_EXECUTE)
                succ_flag = True
            except StopExecutingException:
                log_warning("stop inference")
            finally:
                self.wait_stream()
            return succ_flag

    def get_gradient(self) -> ModelGradient:
        with self.hook_config:
            self.hook_config.disable_log()
            succ: bool = self.inference(
                evaluation_mode=EvaluationMode.TestWithGrad,
            )
            assert succ
            return self.model_util.get_gradients()

    def get_sample_loss(self, evaluation_mode=EvaluationMode.Test) -> dict:
        sample_loss: dict = {}
        with self.hook_config:
            self.hook_config.disable_log()
            hook_name = "__cyy_collect_sample_loss"
            self.append_named_hook(
                hook_point=ExecutorHookPoint.AFTER_BATCH,
                name=hook_name,
                fun=functools.partial(self.__collect_sample_loss, sample_loss),
            )
            evaluation_kwargs = {
                "reduce_loss": False,
                "need_sample_indices": True,
            }
            self.running_model_evaluator.add_evaluation_kwargs(**evaluation_kwargs)
            try:
                succ: bool = self.inference(evaluation_mode=evaluation_mode)
                assert succ
            finally:
                self.running_model_evaluator.remove_evaluation_kwargs(
                    evaluation_kwargs.keys()
                )
                self.remove_named_hook(name=hook_name)
            assert len(sample_loss) == self.dataset_size
            return sample_loss

    def __collect_sample_loss(
        self, sample_loss: dict, result, sample_indices, **kwargs
    ) -> None:
        assert not result["is_averaged_loss"]
        if isinstance(sample_indices, torch.Tensor):
            sample_indices = sample_indices.tolist()
        sample_loss.update(zip(sample_indices, result["loss"]))
