import functools
from collections.abc import Callable, Iterable
from typing import Any

import torch
from cyy_naive_lib.log import log_warning

from .executor import Executor
from .ml_type import (
    EvaluationMode,
    ExecutorHookPoint,
    ModelGradient,
    StopExecutingException,
)


class Inferencer(Executor):
    def inference(
        self,
        evaluation_mode: EvaluationMode = EvaluationMode.Test,
    ) -> bool:
        succ_flag: bool = False
        require_grad: bool = EvaluationMode not in (
            EvaluationMode.Test,
            EvaluationMode.SampleInference,
        )
        with (
            self.complete_stream_context,
            torch.enable_grad() if require_grad else torch.inference_mode(),
        ):
            try:
                self._prepare_execution()
                self._execute_epoch(epoch=1, evaluation_mode=evaluation_mode)
                self.exec_hooks(hook_point=ExecutorHookPoint.AFTER_EXECUTE)
                succ_flag = True
            except StopExecutingException:
                log_warning("stop inference")
            return succ_flag

    def get_gradient(self) -> ModelGradient:
        with self.hook_config:
            self.hook_config.disable_log()
            self.hook_config.use_performance_metric = False
            succ: bool = self.inference(
                evaluation_mode=EvaluationMode.TestWithGrad,
            )
            assert succ
            return self.model_util.get_gradients()

    def get_sample_loss(
        self, evaluation_mode=EvaluationMode.SampleInference
    ) -> dict[int, Any]:
        evaluation_kwargs = {
            "reduce_loss": False,
            "need_sample_indices": True,
        }
        return self._get_sample_output(
            evaluation_mode=evaluation_mode,
            evaluation_kwargs=evaluation_kwargs,
            hook=self.__collect_sample_loss,
        )

    def get_sample_output(self, **generate_kwargs: Any) -> dict[int, Any]:
        evaluation_kwargs = {}
        if generate_kwargs:
            evaluation_kwargs |= {
                "generate": True,
                "generate_kwargs": generate_kwargs,
            }
        return self._get_sample_output(
            evaluation_mode=EvaluationMode.SampleInference,
            evaluation_kwargs=evaluation_kwargs,
            hook=self.__collect_sample_output,
        )

    def __collect_sample_loss(
        self,
        sample_loss: dict[int, Any],
        result: dict,
        sample_indices: Iterable[int],
        **kwargs: Any,
    ) -> None:
        assert not result["is_averaged_loss"]
        if isinstance(sample_indices, torch.Tensor):
            sample_indices = sample_indices.tolist()
        sample_loss.update(zip(sample_indices, result["loss"], strict=False))

    def __collect_sample_output(
        self,
        sample_output: dict[int, Any],
        result: dict,
        sample_indices: Iterable[int],
        **kwargs: Any,
    ) -> None:
        if isinstance(sample_indices, torch.Tensor):
            sample_indices = sample_indices.tolist()
        sample_output.update(zip(sample_indices, result["output"], strict=False))

    def _get_sample_output(
        self, evaluation_mode: EvaluationMode, evaluation_kwargs: dict, hook: Callable
    ) -> dict[int, Any]:
        result: dict[int, Any] = {}
        with self.hook_config:
            self.hook_config.disable_log()
            self.hook_config.use_performance_metric = False
            hook_name = "__collect_sample_output"
            self.append_named_hook(
                hook_point=ExecutorHookPoint.AFTER_BATCH,
                name=hook_name,
                fun=functools.partial(hook, result),
            )
            self.running_model_evaluator.add_evaluation_kwargs(**evaluation_kwargs)
            try:
                succ: bool = self.inference(evaluation_mode=evaluation_mode)
                assert succ
            finally:
                self.running_model_evaluator.remove_evaluation_kwargs(
                    evaluation_kwargs.keys()
                )
                self.remove_named_hook(name=hook_name)
            assert len(result) == self.dataset_size
            return result
