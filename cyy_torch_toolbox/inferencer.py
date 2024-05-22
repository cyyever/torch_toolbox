import asyncio
import functools

import torch
from cyy_naive_lib.concurrency import ThreadPool
from cyy_naive_lib.log import log_warning

from .executor import Executor
from .ml_type import EvaluationMode, ExecutorHookPoint, StopExecutingException
from .typing import ModelGradient


class Inferencer(Executor):
    def inference(self, evaluation_mode: EvaluationMode = EvaluationMode.Test) -> bool:
        co = self.async_inference(evaluation_mode=evaluation_mode)
        try:
            return asyncio.run(co)
        except BaseException as e:
            if "a running event loop" not in str(e):
                raise e
            self.wait_stream()
            pool = ThreadPool()
            pool.submit(asyncio.run, co)
            done, _ = pool.wait_results()
            pool.shutdown()
            assert done
            return list(done.values())[0]

    async def async_inference(
        self,
        evaluation_mode: EvaluationMode = EvaluationMode.Test,
    ) -> bool:
        succ_flag: bool = False
        require_grad: bool = EvaluationMode != EvaluationMode.Test
        with (
            torch.set_grad_enabled(require_grad),
            self.device_context,
            self.device_stream_context,
        ):
            try:
                await self._prepare_execution()
                await self._execute_epoch(epoch=1, evaluation_mode=evaluation_mode)
                await self.async_exec_hooks(hook_point=ExecutorHookPoint.AFTER_EXECUTE)
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
            hook_name = "__cyy_collect_sample_loss"
            self.append_named_hook(
                hook_point=ExecutorHookPoint.AFTER_BATCH,
                name=hook_name,
                fun=functools.partial(self.__collect_sample_loss, sample_loss),
            )
            self.hook_config.disable_log()
            evaluation_kwargs = {
                "reduce_loss": False,
                "need_sample_indices": True,
            }
            self.running_model_evaluator.add_evaluation_kwargs(**evaluation_kwargs)
            succ: bool = self.inference(evaluation_mode=evaluation_mode)
            self.running_model_evaluator.remove_evaluation_kwargs(
                evaluation_kwargs.keys()
            )
            self.remove_named_hook(name=hook_name)
            assert succ
            assert len(sample_loss) == self.dataset_size
            return sample_loss

    def __collect_sample_loss(
        self, sample_loss: dict, result, sample_indices, **kwargs
    ) -> None:
        assert not result["is_averaged_loss"]
        sample_loss.update(zip(sample_indices.tolist(), result["loss"]))
