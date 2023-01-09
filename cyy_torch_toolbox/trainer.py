import contextlib

import torch
from cyy_naive_lib.log import get_logger

from cyy_torch_toolbox.classification_inferencer import \
    ClassificationInferencer
from cyy_torch_toolbox.dataset_collection import DatasetCollection
from cyy_torch_toolbox.hooks.keep_model import KeepModelHook
from cyy_torch_toolbox.hooks.learning_rate_hook import LearningRateHook
from cyy_torch_toolbox.hooks.trainer_debugger import TrainerDebugger
from cyy_torch_toolbox.hyper_parameter import HyperParameter
from cyy_torch_toolbox.inferencer import Inferencer
from cyy_torch_toolbox.metric_visualizers.batch_loss_logger import \
    BatchLossLogger
from cyy_torch_toolbox.ml_type import (MachineLearningPhase,
                                       ModelExecutorHookPoint, ModelType,
                                       StopExecutingException)
from cyy_torch_toolbox.model_executor import ModelExecutor
from cyy_torch_toolbox.model_with_loss import (ModelWithLoss,
                                               ParallelModelWithLoss)


class Trainer(ModelExecutor):
    def __init__(
        self,
        model_with_loss: ModelWithLoss,
        dataset_collection: DatasetCollection,
        hyper_parameter: HyperParameter,
    ):
        super().__init__(
            model_with_loss,
            dataset_collection,
            MachineLearningPhase.Training,
        )
        self.__hyper_parameter = hyper_parameter
        self.__inferencers: dict = {}
        self.append_hook(LearningRateHook(), "learning_rate_hook")
        self.append_hook(BatchLossLogger(), "batch_loss_logger")
        self.append_hook(KeepModelHook(), "keep_model_hook")

    @property
    def hyper_parameter(self):
        return self.__hyper_parameter

    def _get_batch_size(self) -> int:
        return self.hyper_parameter.batch_size

    def set_device(self, device) -> None:
        super().set_device(device)
        for inferencer in self.__inferencers.values():
            inferencer.set_device(device)

    @property
    def best_model(self):
        keep_model_hook = self.get_hook("keep_model_hook")
        if keep_model_hook.best_model is None:
            return None
        return keep_model_hook.best_model[0]

    def get_inferencer_performance_metric(self, phase):
        return self.__inferencers[phase].performance_metric

    def get_inferencer(
        self, phase: MachineLearningPhase, copy_model: bool = False
    ) -> Inferencer:
        model_with_loss = self.copy_model_with_loss(deepcopy=copy_model)

        inferencer = None
        if model_with_loss.model_type == ModelType.Classification:
            inferencer = ClassificationInferencer(
                model_with_loss,
                self.dataset_collection,
                phase=phase,
                batch_size=self._get_batch_size(),
            )
        if inferencer is None:
            raise RuntimeError(
                "Unsupported model type:" + str(model_with_loss.model_type)
            )
        inferencer.cache_transforms = self.cache_transforms
        inferencer.set_device(self.device)
        if self.has_amp():
            inferencer.set_amp()
        return inferencer

    def get_optimizer(self):
        if "optimizer" not in self._data:
            self._data["optimizer"] = self.hyper_parameter.get_optimizer(self)
        return self._data["optimizer"]

    def remove_optimizer(self):
        self._data.pop("optimizer", None)
        self.remove_lr_scheduler()

    def get_lr_scheduler(self):
        if "lr_scheduler" not in self._data:
            self._data["lr_scheduler"] = self.hyper_parameter.get_lr_scheduler(self)
        return self._data["lr_scheduler"]

    def remove_lr_scheduler(self):
        self._data.pop("lr_scheduler", None)

    def load_model(self, model_path):
        super().load_model(model_path)
        self.remove_optimizer()

    def load_parameter_dict(self, parameter_dict: dict) -> None:
        self.model_util.load_parameter_dict(parameter_dict)
        self.remove_optimizer()

    def offload_from_gpu(self):
        self.__inferencers.clear()
        super().offload_from_gpu()

    def offload_from_memory(self):
        super().offload_from_memory()
        self.get_hook("keep_model_hook").offload_from_memory()

    def _prepare_execution(
        self,
        use_DDP: bool = False,
        save_best_model: bool = False,
        save_epoch_model: bool = False,
        save_last_model: bool = False,
        batch_loss_log_times: None | int = None,
        **kwargs: dict
    ) -> None:
        keep_model_hook = self.get_hook("keep_model_hook")
        keep_model_hook.save_best_model = save_best_model
        keep_model_hook.save_epoch_model = save_epoch_model
        keep_model_hook.save_last_model = save_last_model
        self.__inferencers.clear()
        if self.debugging_mode:
            get_logger().warning("train in debugging mode")
            if not self.has_hook_obj("debugger"):
                self.append_hook(TrainerDebugger(), "debugger")
            else:
                self.enable_hook(hook_name="debugger")
        else:
            if self.has_hook_obj("debugger"):
                self.disable_hook(hook_name="debugger")
        if batch_loss_log_times is not None:
            self.get_hook("batch_loss_logger").log_times = batch_loss_log_times
        if use_DDP:
            self._model_with_loss = ParallelModelWithLoss.create(self._model_with_loss)
        super()._prepare_execution(**kwargs)

    def train(self, **kwargs):
        with (
            torch.cuda.device(self.device)
            if self.cuda_stream is not None
            else contextlib.nullcontext(),
            torch.cuda.stream(self.cuda_stream),
        ):
            self._prepare_execution(**kwargs)
            try:
                for epoch in range(1, self.hyper_parameter.epoch + 1):
                    self._execute_epoch(epoch=epoch)

                    for phase in (
                        MachineLearningPhase.Validation,
                        MachineLearningPhase.Test,
                    ):
                        if (
                            phase not in self.__inferencers
                            and self.dataset_collection.has_dataset(phase=phase)
                        ):
                            inferencer = self.get_inferencer(phase)
                            inferencer.disable_hook("logger")
                            self.__inferencers[phase] = inferencer
                        inferencer = self.__inferencers.get(phase, None)
                        if inferencer is not None:
                            inferencer.model.load_state_dict(self.model.state_dict())
                            inferencer.inference(epoch=epoch, use_grad=False)

                    self.exec_hooks(
                        ModelExecutorHookPoint.AFTER_VALIDATION,
                        epoch=epoch,
                    )
                    # adjust learning rate
                    lr_scheduler = self.get_lr_scheduler()
                    if HyperParameter.lr_scheduler_step_after_batch(lr_scheduler):
                        continue
                    match lr_scheduler:
                        case torch.optim.lr_scheduler.ReduceLROnPlateau():
                            training_loss = self.get_hook(
                                "performance_metric"
                            ).get_loss(epoch)
                            get_logger().debug(
                                "call ReduceLROnPlateau for training loss %s",
                                training_loss,
                            )
                            lr_scheduler.step(training_loss)
                        case _:
                            lr_scheduler.step()
            except StopExecutingException:
                get_logger().warning("stop training")
            finally:
                self._wait_stream()
        self.exec_hooks(
            ModelExecutorHookPoint.AFTER_EXECUTE,
            epoch=self.hyper_parameter.epoch,
        )

    def _get_backward_loss(self, result):
        return result["loss"]
