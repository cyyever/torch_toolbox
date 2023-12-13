import abc
import contextlib
import copy
import os
from typing import Any, Callable, Generator

import torch
import torch.utils.data
from cyy_naive_lib.log import get_logger

from .data_pipeline.loader import get_dataloader
from .dataset import DatasetCollection, DatasetUtil
from .device import get_device
from .hook import HookCollection
from .hook.config import HookConfig
from .hyper_parameter import HyperParameter, lr_scheduler_step_after_batch
from .metric_visualizers.metric_visualizer import MetricVisualizer
from .metrics.performance_metric import PerformanceMetric
from .ml_type import EvaluationMode, ExecutorHookPoint, MachineLearningPhase
from .model import ModelEvaluator, ModelUtil


class Executor(HookCollection, abc.ABC):
    def __init__(
        self,
        model_evaluator: ModelEvaluator,
        dataset_collection: DatasetCollection,
        phase: MachineLearningPhase,
        hyper_parameter: HyperParameter,
        hook_config: HookConfig | None = None,
    ) -> None:
        super().__init__()
        self._data: dict = {}
        self.__model_evaluator: ModelEvaluator = model_evaluator
        self.__dataset_collection: DatasetCollection = dataset_collection
        self.__phase: MachineLearningPhase = phase
        self.__hyper_parameters: dict = {phase: copy.deepcopy(hyper_parameter)}
        if not hook_config:
            hook_config = HookConfig()
        self.hook_config: HookConfig = hook_config
        self.__device: None | torch.device = None
        self.__dataloader: None | torch.utils.data.DataLoader = None
        self.__dataloader_kwargs: dict = {}
        self.__device_stream: None | torch._C._CudaStreamBase = None
        self.__save_dir: None | str = None
        self.cache_transforms: None | str = "cpu"

    @property
    def dataset_collection(self) -> DatasetCollection:
        return self.__dataset_collection

    @property
    def device(self) -> torch.device:
        if self.__device is None:
            self.set_device(get_device())
        assert self.__device is not None
        return self.__device

    @property
    def hyper_parameter(self) -> HyperParameter:
        return self.__hyper_parameters[self.phase]

    def set_phase(self, phase: MachineLearningPhase):
        self.__phase = phase

    def set_hyper_parameter(
        self, hyper_parameter: HyperParameter, phase: MachineLearningPhase | None = None
    ) -> None:
        if phase is None:
            phase = self.phase
        self.__hyper_parameters[phase] = hyper_parameter

    @property
    def performance_metric(self) -> PerformanceMetric:
        hook = self.get_hook("performance_metric")
        assert isinstance(hook, PerformanceMetric)
        return hook

    @property
    def phase(self) -> MachineLearningPhase:
        return self.__phase

    def exec_hooks(self, hook_point: ExecutorHookPoint, **kwargs: Any) -> None:
        kwargs["executor"] = self
        super().exec_hooks(hook_point=hook_point, **kwargs)

    def set_save_dir(self, save_dir: str) -> None:
        self.__save_dir = save_dir
        data_dir = os.path.join(save_dir, "visualizer")
        for hook in self._hooks.values():
            if isinstance(hook, MetricVisualizer):
                hook.set_data_dir(data_dir)
        for executor in self._foreach_sub_executor():
            executor.set_save_dir(save_dir)

    def set_visualizer_prefix(self, prefix: str) -> None:
        for hook in self._hooks.values():
            if isinstance(hook, MetricVisualizer):
                hook.set_prefix(prefix)

    @property
    def visualizer_prefix(self) -> None | str:
        for hook in self._hooks.values():
            if isinstance(hook, MetricVisualizer):
                return hook.prefix
        return None

    @property
    def save_dir(self) -> None | str:
        return self.__save_dir

    @property
    def dataset(self):
        return self.dataset_collection.get_dataset(phase=self.__phase)

    @property
    def dataset_util(self) -> DatasetUtil:
        return self.dataset_collection.get_dataset_util(phase=self.__phase)

    @property
    def dataset_size(self) -> int:
        if "dataset_size" not in self._data:
            self.__refresh_dataset_size()
        return self._data["dataset_size"]

    def __refresh_dataset_size(self) -> None:
        self._data["dataset_size"] = len(self.dataset_util)

    def update_dataloader_kwargs(self, **kwargs: Any) -> None:
        self.__dataloader_kwargs.update(kwargs)
        self.__dataloader = None

    @property
    def dataloader(self) -> torch.utils.data.DataLoader:
        if self.__dataloader is None:
            self.__dataloader = get_dataloader(
                dc=self.dataset_collection,
                phase=self.__phase,
                hyper_parameter=self.hyper_parameter,
                device=self.device,
                model_evaluator=self.running_model_evaluator,
                cache_transforms=self.cache_transforms,
                **self.__dataloader_kwargs,
            )
        return self.__dataloader

    @property
    def running_model_evaluator(self) -> ModelEvaluator:
        return self.__model_evaluator

    @property
    def model_evaluator(self) -> ModelEvaluator:
        self.wait_stream()
        return self.__model_evaluator

    @property
    def model_util(self) -> ModelUtil:
        return self.running_model_evaluator.model_util

    @property
    def loss_fun(self) -> Callable:
        return self.__model_evaluator.loss_fun

    @property
    def model(self) -> torch.nn.Module:
        return self.running_model_evaluator.model

    def replace_model(self, fun: Callable) -> None:
        self.__model_evaluator = self.model_evaluator.replace_model(fun(self.model))

    def _prepare_execution(self) -> None:
        self._data.clear()
        self.hook_config.set_hooks(self)
        self.exec_hooks(hook_point=ExecutorHookPoint.BEFORE_EXECUTE)

    def set_device(self, device: torch.device) -> None:
        if self.__device != device:
            self.wait_stream()
            self.__device = device
            get_logger().debug("%s use device %s", str(self.__phase), self.__device)
            self.__device_stream = None
            self.__dataloader = None

        for executor in self._foreach_sub_executor():
            executor.set_device(device)

    def __getstate__(self):
        # capture what is normally pickled
        state = self.__dict__.copy()
        state["_Executor__device"] = None
        state["_Executor__device_stream"] = None
        state["_Executor__dataloader"] = None
        return state

    @property
    def device_context(self) -> Any:
        return (
            contextlib.nullcontext()
            if "cuda" not in self.device.type.lower() or not torch.cuda.is_available()
            else torch.cuda.device(self.device)
        )

    @property
    def device_stream_context(self) -> torch.cuda.StreamContext:
        if "cuda" in self.device.type.lower():
            if self.__device_stream is None:
                self.__device_stream = torch.cuda.Stream(device=self.device)
                self.__device_stream.wait_stream(torch.cuda.current_stream())
            return torch.cuda.stream(self.__device_stream)
        return torch.cuda.stream(None)

    def wait_stream(self) -> None:
        if self.__device_stream is not None:
            self.__device_stream.synchronize()
            assert self.__device_stream.query()

    def set_dataset_collection(self, dc: DatasetCollection) -> None:
        self.wait_stream()
        self.__dataset_collection = dc

    def set_model_evaluator(self, model_evaluator: ModelEvaluator) -> None:
        self.wait_stream()
        self.__model_evaluator = model_evaluator

    def load_model(self, model_path: str) -> None:
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def _foreach_sub_executor(self) -> Generator:
        yield from []

    def save_model(self, model_path: str) -> None:
        torch.save(self.model.state_dict(), model_path)

    def offload_from_device(self) -> None:
        self.wait_stream()
        self.__model_evaluator.offload_from_device()
        torch.cuda.empty_cache()
        for executor in self._foreach_sub_executor():
            executor.offload_from_device()

    def has_optimizer(self) -> bool:
        return "optimizer" in self._data

    def has_lr_scheduler(self) -> bool:
        return "lr_scheduler" in self._data

    def get_optimizer(self) -> torch.optim.Optimizer:
        if "optimizer" not in self._data:
            self._data["optimizer"] = self.hyper_parameter.get_optimizer(self)
        return self._data["optimizer"]

    def get_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        if "lr_scheduler" not in self._data:
            self._data["lr_scheduler"] = self.hyper_parameter.get_lr_scheduler(self)
        return self._data["lr_scheduler"]

    def get_forward_context(self):
        if not self._data["forward_contexts"]:
            return contextlib.nullcontext()
        assert len(self._data["forward_contexts"]) == 1
        return self._data["forward_contexts"][0]

    def execute_batch(
        self,
        batch_index: int,
        batch: Any,
        epoch: int,
        evaluation_mode: EvaluationMode,
    ) -> None:
        self.exec_hooks(
            hook_point=ExecutorHookPoint.AFTER_FETCH_BATCH,
            batch_index=batch_index,
        )
        batch |= {
            "batch_index": batch_index,
            "phase": self.phase,
            "device": self.device,
            "evaluation_mode": evaluation_mode,
            "non_blocking": True,
        }
        if (
            evaluation_mode == EvaluationMode.Training
            and self.hyper_parameter.batch_size != 1
            and batch.get("batch_size", None) == 1
            and self.__model_evaluator.model_util.have_module(
                module_type=torch.nn.BatchNorm2d
            )
        ):
            get_logger().debug("drop last one-sized batch for batch norm")
            return None

        self._data["forward_contexts"] = []
        self.exec_hooks(
            hook_point=ExecutorHookPoint.BEFORE_BATCH,
            epoch=epoch,
            **batch,
        )

        forward_result: dict = {}

        with self.get_forward_context():
            evaluation_kwargs = batch
            if self.has_hook(ExecutorHookPoint.MODEL_FORWARD):
                self.exec_hooks(
                    hook_point=ExecutorHookPoint.MODEL_FORWARD,
                    evaluation_kwargs=evaluation_kwargs,
                )
                forward_result = self._data.pop("forward_result")
            else:
                forward_result = self.__model_evaluator(**evaluation_kwargs)

            if forward_result["is_averaged_loss"]:
                assert self.dataset_size > 1
                forward_result["normalized_batch_loss"] = (
                    forward_result["loss"]
                    * forward_result["loss_batch_size"]
                    / self.dataset_size
                )

            batch |= forward_result
        if evaluation_mode == EvaluationMode.Training:
            optimizer: torch.optim.Optimizer = self.get_optimizer()

        if evaluation_mode != EvaluationMode.Test:
            loss = self._get_backward_loss(result=forward_result)
            assert loss is not None
            if self.has_hook(ExecutorHookPoint.MODEL_BACKWARD):
                self.exec_hooks(hook_point=ExecutorHookPoint.MODEL_BACKWARD, loss=loss)
            else:
                if evaluation_mode == EvaluationMode.TestWithGrad:
                    self.running_model_evaluator.model.zero_grad(set_to_none=True)
                elif evaluation_mode == EvaluationMode.Training:
                    optimizer.zero_grad(set_to_none=True)
                loss.backward()

        if evaluation_mode == EvaluationMode.Training:
            if self.has_hook(ExecutorHookPoint.OPTIMIZER_STEP):
                self.exec_hooks(ExecutorHookPoint.OPTIMIZER_STEP, optimizer=optimizer)
            else:
                optimizer.step()
            lr_scheduler = self.get_lr_scheduler()
            if lr_scheduler_step_after_batch(lr_scheduler):
                get_logger().debug("adjust lr after batch")
                lr_scheduler.step()

        self.exec_hooks(
            hook_point=ExecutorHookPoint.AFTER_BATCH,
            epoch=epoch,
            result=forward_result,
            **batch,
        )

    def _execute_epoch(
        self,
        epoch: int,
        evaluation_mode: EvaluationMode,
    ) -> None:
        self.exec_hooks(
            hook_point=ExecutorHookPoint.BEFORE_EPOCH,
            epoch=epoch,
        )
        self.__refresh_dataset_size()
        self.exec_hooks(hook_point=ExecutorHookPoint.BEFORE_FETCH_BATCH, batch_index=0)
        for batch_index, batch in enumerate(self.dataloader):
            self.exec_hooks(
                hook_point=ExecutorHookPoint.BEFORE_FETCH_BATCH, batch_index=0
            )
            self.execute_batch(
                batch_index=batch_index,
                batch=batch,
                epoch=epoch,
                evaluation_mode=evaluation_mode,
            )
            self.exec_hooks(
                hook_point=ExecutorHookPoint.BEFORE_FETCH_BATCH,
                batch_index=batch_index + 1,
            )
        if evaluation_mode == EvaluationMode.Training:
            lr_scheduler = self.get_lr_scheduler()
            if not lr_scheduler_step_after_batch(lr_scheduler):
                match lr_scheduler:
                    case torch.optim.lr_scheduler.ReduceLROnPlateau():
                        training_loss = self.performance_metric.get_loss(epoch)
                        get_logger().debug(
                            "call ReduceLROnPlateau for training loss %s",
                            training_loss,
                        )
                        lr_scheduler.step(training_loss)
                    case _:
                        lr_scheduler.step()

        self.exec_hooks(
            hook_point=ExecutorHookPoint.AFTER_EPOCH,
            epoch=epoch,
        )

    def _get_backward_loss(self, result):
        raise NotImplementedError()
