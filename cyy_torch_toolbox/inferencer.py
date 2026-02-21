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
        require_grad: bool = evaluation_mode not in (
            EvaluationMode.Test,
            EvaluationMode.SampleInference,
        )
        with (
            self.stream_context,
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
        self, evaluation_mode: EvaluationMode = EvaluationMode.SampleInference
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

    def process_sample_output(
        self, callback: Callable, **generate_kwargs: Any
    ) -> dict[int, Any]:
        evaluation_kwargs: dict[str, Any] = {}
        if generate_kwargs:
            evaluation_kwargs |= {
                "generate": True,
                "generate_kwargs": generate_kwargs,
            }
        return self._get_sample_output(
            evaluation_mode=EvaluationMode.SampleInference,
            evaluation_kwargs=evaluation_kwargs,
            hook=functools.partial(self.__process_sample_output, callback=callback),
        )

    def __collect_sample_loss(
        self,
        final_result: dict[int, Any],
        result: dict[str, Any],
        sample_indices: Iterable[int],
        **kwargs: Any,
    ) -> None:
        assert not result["is_averaged_loss"]
        if isinstance(sample_indices, torch.Tensor):
            sample_indices = sample_indices.tolist()
        final_result.update(zip(sample_indices, result["loss"], strict=False))

    def __process_sample_output(
        self,
        callback: Callable,
        final_result: dict[int, Any],
        **kwargs: Any,
    ) -> None:
        callback(kwargs)

    def _get_sample_output(
        self,
        evaluation_mode: EvaluationMode,
        evaluation_kwargs: dict[str, Any],
        hook: Callable,
    ) -> dict[int, Any]:
        result: dict[int, Any] = {}
        with self.hook_config:
            self.hook_config.disable_log()
            self.hook_config.use_performance_metric = False
            hook_name = "__get_sample_output"
            self.append_named_hook(
                hook_point=ExecutorHookPoint.AFTER_BATCH,
                name=hook_name,
                fun=functools.partial(hook, final_result=result),
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
            if result:
                assert len(result) == self.dataset_size
            return result
