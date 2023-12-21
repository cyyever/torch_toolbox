import copy
from typing import Any, Generator

import torch
from cyy_naive_lib.log import get_logger

from .classification_inferencer import ClassificationInferencer
from .dataset import DatasetCollection
from .executor import Executor
from .hook.config import HookConfig
from .hyper_parameter import HyperParameter
from .inferencer import Inferencer
from .metric_visualizers.batch_loss_logger import BatchLossLogger
from .ml_type import (EvaluationMode, ExecutorHookPoint, MachineLearningPhase,
                      ModelType, StopExecutingException)
from .model import ModelEvaluator


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
        self.__optimizer_parameters = None
        self.append_hook(BatchLossLogger(), "batch_loss_logger")

    def __getstate__(self):
        # capture what is normally pickled
        state = super().__getstate__()
        state["_Trainer__inferencers"] = {}
        state["_Trainer__optimizer_parameters"] = None
        return state

    def get_cached_inferencer(self, phase: MachineLearningPhase) -> Inferencer | None:
        return self.__inferencers.get(phase, None)

    def get_inferencer(
        self, phase: MachineLearningPhase, deepcopy_model: bool = False
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
                hook_config=copy.copy(self.hook_config),
            )
        if inferencer is None:
            raise RuntimeError(
                "Unsupported model type:" + str(model_evaluator.model_type)
            )
        inferencer.cache_transforms = self.cache_transforms
        inferencer.set_device(self.device)
        if self.save_dir is not None:
            inferencer.set_save_dir(self.save_dir)
        if self.visualizer_prefix is not None:
            inferencer.set_visualizer_prefix(self.visualizer_prefix)
        return inferencer

    def reset_optimizer_parameters(self, parameters: Any) -> None:
        self.__optimizer_parameters = parameters
        self.remove_optimizer()

    def get_optimizer(self) -> torch.optim.Optimizer:
        if "optimizer" not in self._data:
            self._data["optimizer"] = self.hyper_parameter.get_optimizer(
                self, parameters=self.__optimizer_parameters
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

    def load_parameter_dict(self, parameter_dict: dict) -> None:
        self.model_util.load_parameter_dict(parameter_dict)
        self.remove_optimizer()

    def train(self, validate: bool = True) -> None:
        with (
            self.device_context,
            self.device_stream_context,
        ):
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
                get_logger().warning("stop training")
                self.exec_hooks(hook_point=ExecutorHookPoint.AFTER_EXECUTE)
            finally:
                self.wait_stream()

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
        inferencer.model.load_state_dict(self.model.state_dict())
        inferencer.inference()
        return True

    def _foreach_sub_executor(self) -> Generator:
        yield from self.__inferencers.values()


class TrainerConfig:
    def __init__(self) -> None:
        self.hook_config: HookConfig = HookConfig()
        self.cache_transforms: None | str = None

    def create_trainer(
        self,
        dataset_collection: DatasetCollection,
        model_evaluator: ModelEvaluator,
        hyper_parameter: HyperParameter,
    ) -> Trainer:
        dataset_collection.add_transforms(
            model_evaluator=model_evaluator,
        )
        trainer = Trainer(
            model_evaluator=model_evaluator,
            dataset_collection=dataset_collection,
            hyper_parameter=hyper_parameter,
            hook_config=self.hook_config,
        )
        trainer.cache_transforms = self.cache_transforms
        return trainer
