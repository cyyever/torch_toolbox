import copy
from typing import Any

import torch
from cyy_naive_lib.log import get_logger

from .classification_inferencer import ClassificationInferencer
from .dataset_collection import DatasetCollection
from .executor import Executor
from .hook.config import HookConfig
from .hook.keep_model import KeepModelHook
from .hyper_parameter import HyperParameter
from .inferencer import Inferencer
from .metric_visualizers.batch_loss_logger import BatchLossLogger
from .ml_type import (ExecutorHookPoint, MachineLearningPhase, ModelType,
                      StopExecutingException)
from .model_evaluator import ModelEvaluator


class Trainer(Executor):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            phase=MachineLearningPhase.Training,
            **kwargs,
        )
        self.__inferencers: dict = {}
        self.append_hook(BatchLossLogger(), "batch_loss_logger")
        self.append_hook(KeepModelHook(), "keep_model_hook")

    def set_device(self, device: torch.device) -> None:
        super().set_device(device)
        for inferencer in self.__inferencers.values():
            inferencer.set_device(device)

    @property
    def best_model(self) -> Any:
        keep_model_hook: KeepModelHook = self.get_hook("keep_model_hook")
        if keep_model_hook.best_model is None:
            return None
        return keep_model_hook.best_model[0]

    @property
    def best_epoch(self) -> int | None:
        keep_model_hook: KeepModelHook = self.get_hook("keep_model_hook")
        if keep_model_hook.best_model is None:
            return None
        return keep_model_hook.best_model[2]

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
                hook_config=self._hook_config,
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

    def get_optimizer(self):
        if "optimizer" not in self._data:
            self._data["optimizer"] = self.hyper_parameter.get_optimizer(self)
        return self._data["optimizer"]

    def remove_optimizer(self) -> None:
        self._data.pop("optimizer", None)
        self.remove_lr_scheduler()

    def get_lr_scheduler(self):
        if "lr_scheduler" not in self._data:
            self._data["lr_scheduler"] = self.hyper_parameter.get_lr_scheduler(self)
        return self._data["lr_scheduler"]

    def remove_lr_scheduler(self) -> None:
        self._data.pop("lr_scheduler", None)

    def load_model(self, model_path: str) -> None:
        super().load_model(model_path)
        self.remove_optimizer()

    def load_parameter_dict(self, parameter_dict: dict) -> None:
        self.model_util.load_parameter_dict(parameter_dict)
        self.remove_optimizer()

    def offload_from_device(self) -> None:
        if self.__inferencers:
            for inferencer in self.__inferencers.values():
                inferencer.offload_from_device()
        super().offload_from_device()

    def _prepare_execution(
        self,
        batch_loss_log_times: None | int = None,
        keep_best_model: bool = False,
        save_best_model: bool = False,
        save_epoch_model: bool = False,
        save_last_model: bool = False,
        **kwargs: Any,
    ) -> None:
        keep_model_hook = self.get_hook("keep_model_hook")
        keep_model_hook.keep_best_model = keep_best_model
        keep_model_hook.save_best_model = save_best_model
        keep_model_hook.save_epoch_model = save_epoch_model
        keep_model_hook.save_last_model = save_last_model
        if batch_loss_log_times is not None:
            self.get_hook("batch_loss_logger").log_times = batch_loss_log_times
        if self.visualizer_prefix is not None and self.__inferencers:
            for inferencer in self.__inferencers.values():
                inferencer.set_visualizer_prefix(self.visualizer_prefix)
        super()._prepare_execution(**kwargs)

    def train(self, run_validation: bool = True, **kwargs: Any) -> None:
        try:
            with (
                self.device_context,
                self.device_stream_context,
            ):
                self._prepare_execution(**kwargs)
                for epoch in range(1, self.hyper_parameter.epoch + 1):
                    self._execute_epoch(epoch=epoch, in_training=True)

                    if run_validation:
                        for phase in (
                            MachineLearningPhase.Validation,
                            MachineLearningPhase.Test,
                        ):
                            if (
                                phase not in self.__inferencers
                                and self.dataset_collection.has_dataset(phase=phase)
                            ):
                                inferencer = self.get_inferencer(
                                    phase=phase, deepcopy_model=False
                                )
                                inferencer.disable_hook("logger")
                                self.__inferencers[phase] = inferencer
                            inferencer = self.__inferencers.get(phase, None)
                            if inferencer is not None:
                                inferencer.model.load_state_dict(
                                    self.model.state_dict()
                                )
                                inferencer.inference(epoch=epoch, use_grad=False)

                            self.exec_hooks(
                                ExecutorHookPoint.AFTER_VALIDATION,
                                epoch=epoch,
                            )
        except StopExecutingException:
            get_logger().warning("stop training")
        finally:
            self.wait_stream()
        self.exec_hooks(
            hook_point=ExecutorHookPoint.AFTER_EXECUTE,
            epoch=self.hyper_parameter.epoch,
        )

    def _get_backward_loss(self, result):
        return result["loss"]


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
