import asyncio
import copy
from typing import Any, Generator

import torch
from cyy_naive_lib.log import log_warning

from .classification_inferencer import ClassificationInferencer
from .dataset import DatasetCollection
from .executor import Executor, ExecutorConfig
from .hyper_parameter import HyperParameter
from .inferencer import Inferencer
from .metric_visualizers import BatchLossLogger
from .ml_type import (EvaluationMode, ExecutorHookPoint, MachineLearningPhase,
                      ModelType, StopExecutingException)
from .model import ModelEvaluator
from .typing import TensorDict


class Trainer(Executor):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            phase=MachineLearningPhase.Training,
            **kwargs,
        )
        self.__inferencers: dict[MachineLearningPhase, Inferencer] = {}
        self.append_hook(BatchLossLogger(), "batch_loss_logger")

    def __getstate__(self) -> dict:
        # capture what is normally pickled
        state = super().__getstate__()
        state["_Trainer__inferencers"] = {}
        state["_Trainer__optimizer_parameters"] = None
        return state

    def get_cached_inferencer(self, phase: MachineLearningPhase) -> Inferencer | None:
        return self.__inferencers.get(phase, None)

    def get_inferencer(
        self,
        phase: MachineLearningPhase,
        deepcopy_model: bool = False,
        inherent_device: bool = True,
    ) -> Inferencer:
        if deepcopy_model:
            model_evaluator: ModelEvaluator = copy.deepcopy(self.model_evaluator)
        else:
            model_evaluator = copy.copy(self.model_evaluator)
        inferencer: Inferencer | None = None
        if model_evaluator.model_type == ModelType.Classification:
            inferencer = ClassificationInferencer(
                model_evaluator,
                self.dataset_collection,
                phase=phase,
                hyper_parameter=self.hyper_parameter,
                hook_config=self.hook_config,
                dataloader_kwargs=self.dataloader_kwargs,
            )
        if inferencer is None:
            raise RuntimeError(
                "Unsupported model type:" + str(model_evaluator.model_type)
            )
        if inherent_device:
            inferencer.set_device(self.device)
        if self.save_dir is not None:
            inferencer.set_save_dir(self.save_dir)
        inferencer.set_visualizer_prefix(self.visualizer_prefix)
        return inferencer

    def reset_optimizer_parameters(self, parameters: Any) -> None:
        self._data["optimizer_parameters"] = parameters
        self.remove_optimizer()

    def get_optimizer(self) -> torch.optim.Optimizer:
        if "optimizer" not in self._data:
            self._data["optimizer"] = self.hyper_parameter.get_optimizer(
                self, parameters=self._data.get("optimizer_parameters", None)
            )
        return self._data["optimizer"]

    def remove_optimizer(self) -> None:
        self._data.pop("optimizer", None)
        self.remove_lr_scheduler()

    def remove_lr_scheduler(self) -> None:
        self._data.pop("lr_scheduler", None)

    def load_model(self, model_path: str) -> None:
        super().load_model(model_path)
        self.remove_optimizer()

    def load_parameter_dict(self, parameter_dict: TensorDict) -> None:
        self.model_util.load_parameter_dict(parameter_dict)
        self.remove_optimizer()

    def train(self, validate: bool = True) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.run_until_complete(self.async_train(validate=validate))
        except BaseException:
            asyncio.run(self.async_train(validate=validate))

    async def async_train(self, validate: bool = True) -> None:
        with (
            self.device_context,
            self.device_stream_context,
        ):
            try:
                await self._prepare_execution()
                for epoch in range(1, self.hyper_parameter.epoch + 1):
                    await self._execute_epoch(
                        epoch=epoch, evaluation_mode=EvaluationMode.Training
                    )
                    if validate and await self.__test(
                        phase=MachineLearningPhase.Validation
                    ):
                        await self.async_exec_hooks(
                            ExecutorHookPoint.AFTER_VALIDATION,
                            epoch=epoch,
                        )
                    await self.__test(phase=MachineLearningPhase.Test)
                await self.async_exec_hooks(hook_point=ExecutorHookPoint.AFTER_EXECUTE)
            except StopExecutingException:
                log_warning("stop training")
                await self.async_exec_hooks(hook_point=ExecutorHookPoint.AFTER_EXECUTE)
            finally:
                self.wait_stream()

    async def __test(self, phase: MachineLearningPhase) -> bool:
        assert phase in (MachineLearningPhase.Validation, MachineLearningPhase.Test)
        if phase not in self.__inferencers and self.dataset_collection.has_dataset(
            phase=phase
        ):
            tmp_inferencer = self.get_inferencer(phase=phase, deepcopy_model=False)
            tmp_inferencer.hook_config.summarize_executor = False
            self.__inferencers[phase] = tmp_inferencer
        inferencer: None | Inferencer = self.__inferencers.get(phase, None)
        if inferencer is None:
            return False
        inferencer.model.load_state_dict(self.model.state_dict())
        inferencer.set_visualizer_prefix(self.visualizer_prefix)
        await inferencer.async_inference()
        return True

    def _foreach_sub_executor(self) -> Generator:
        yield from self.__inferencers.values()


class TrainerConfig(ExecutorConfig):
    def create_trainer(
        self,
        dataset_collection: DatasetCollection,
        model_evaluator: ModelEvaluator,
        hyper_parameter: HyperParameter,
    ) -> Trainer:
        return self.create_executor(
            cls=Trainer,
            dataset_collection=dataset_collection,
            model_evaluator=model_evaluator,
            hyper_parameter=hyper_parameter,
        )
