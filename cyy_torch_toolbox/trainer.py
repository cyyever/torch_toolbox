import copy
from collections.abc import Generator
from typing import Any

import torch
from cyy_naive_lib.log import log_debug, log_warning

from .dataset import DatasetCollectionConfig
from .executor import Executor, ExecutorConfig
from .hyper_parameter import HyperParameter
from .inferencer import Inferencer
from .metric_visualizers import BatchLossLogger
from .ml_type import (
    EvaluationMode,
    ExecutorHookPoint,
    MachineLearningPhase,
    ModelParameter,
    StopExecutingException,
)
from .model import ModelConfig, ModelEvaluator


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
        copy_model: bool = True,
        copy_dataset: bool = False,
        inherent_device: bool = True,
    ) -> Inferencer:
        inferencer = Inferencer(
            dataset_collection_config=self.dataset_collection_config,
            phase=phase,
            hyper_parameter=self.hyper_parameter,
            hook_config=self.hook_config,
            dataloader_kwargs=self.dataloader_kwargs,
            model_config=self.mutable_model_config if not copy_model else None,
        )
        if copy_dataset:
            inferencer.set_dataset_collection(copy.copy(self.dataset_collection))
        if copy_model:
            if deepcopy_model:
                model_evaluator: ModelEvaluator = copy.deepcopy(self.model_evaluator)
            else:
                model_evaluator = copy.copy(self.model_evaluator)
            inferencer.set_model_evaluator(model_evaluator=model_evaluator)
        if inherent_device and self.has_device():
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

    def remove_model(self, remove_optimizer: bool = True) -> None:
        self.__inferencers.clear()
        if remove_optimizer:
            self.remove_optimizer()
        super().remove_model()

    def remove_optimizer(self) -> None:
        self._data.pop("optimizer", None)
        self.remove_lr_scheduler()

    def remove_lr_scheduler(self) -> None:
        self._data.pop("lr_scheduler", None)

    def load_parameters(self, parameter: ModelParameter) -> None:
        self.model_util.load_parameters(parameter)
        self.remove_optimizer()

    def train(self, validate: bool = True) -> None:
        with self.complete_stream_context:
            try:
                self._prepare_execution()
                for epoch in range(1, self.hyper_parameter.epoch + 1):
                    self._execute_epoch(
                        epoch=epoch, evaluation_mode=EvaluationMode.Training
                    )
                    if validate and self.__test(phase=MachineLearningPhase.Validation):
                        self.exec_hooks(
                            ExecutorHookPoint.AFTER_VALIDATION,
                            epoch=epoch,
                        )
                    self.__test(phase=MachineLearningPhase.Test)
                self.exec_hooks(hook_point=ExecutorHookPoint.AFTER_EXECUTE)
            except StopExecutingException:
                log_warning("stop training")
                self.exec_hooks(hook_point=ExecutorHookPoint.AFTER_EXECUTE)

    def _execute_epoch(
        self,
        epoch: int,
        evaluation_mode: EvaluationMode,
    ) -> None:
        super()._execute_epoch(epoch=epoch, evaluation_mode=evaluation_mode)
        if evaluation_mode == EvaluationMode.Training:
            lr_scheduler = self.get_lr_scheduler()
            match lr_scheduler:
                case torch.optim.lr_scheduler.ReduceLROnPlateau():
                    inferencer = self.get_cached_inferencer(
                        MachineLearningPhase.Validation
                    )
                    loss = None
                    if inferencer is not None:
                        loss = inferencer.performance_metric.get_loss(1, to_item=False)
                    else:
                        loss = self.performance_metric.get_loss(epoch, to_item=False)
                    assert loss is not None
                    log_debug(
                        "call ReduceLROnPlateau for loss %s",
                        loss,
                    )
                    lr_scheduler.step(loss)

    def __test(self, phase: MachineLearningPhase) -> bool:
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
        inferencer.model_evaluator.load_model_for_inference(self.model)
        inferencer.set_visualizer_prefix(self.visualizer_prefix)
        inferencer.inference()
        return True

    def _foreach_sub_executor(self) -> Generator:
        yield from self.__inferencers.values()


class TrainerConfig(ExecutorConfig):
    def create_trainer(
        self,
        dataset_collection_config: DatasetCollectionConfig,
        model_config: ModelConfig,
        hyper_parameter: HyperParameter,
    ) -> Trainer:
        return self.create_executor(
            cls=Trainer,
            dataset_collection_config=dataset_collection_config,
            model_config=model_config,
            hyper_parameter=hyper_parameter,
        )
