import abc
import copy
import functools
import gc
import os
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.cuda
import torch.utils.data
from cyy_naive_lib.log import log_debug

from .data_pipeline.loader import get_dataloader
from .dataset import DatasetCollection, DatasetCollectionConfig, DatasetUtil
from .device import DefaultDeviceContext, DeviceGreedyAllocator, SyncedStreamContext
from .hook import HookCollection
from .hook.config import HookConfig
from .hyper_parameter import HyperParameter, lr_scheduler_step_after_batch
from .metric_visualizers import MetricVisualizer
from .metrics import PerformanceMetric
from .ml_type import ConfigBase, EvaluationMode, ExecutorHookPoint, MachineLearningPhase
from .model import ModelConfig, ModelEvaluator, ModelUtil


class Executor(HookCollection, abc.ABC):
    def __init__(
        self,
        dataset_collection_config: DatasetCollectionConfig,
        phase: MachineLearningPhase,
        hyper_parameter: HyperParameter,
        model_config: ModelConfig | None = None,
        hook_config: HookConfig | None = None,
        dataloader_kwargs: dict | None = None,
        auto_create_model: bool = True,
    ) -> None:
        super().__init__()
        self._data: dict = {}
        self.__dataset_collection_config: DatasetCollectionConfig = copy.deepcopy(
            dataset_collection_config
        )
        self.__dataset_collection: DatasetCollection | None = None
        self.__model_config = copy.deepcopy(model_config)
        self.__auto_create_model = auto_create_model
        self.__model_evaluator: ModelEvaluator | None = None
        self.__phase: MachineLearningPhase = phase
        self.__hyper_parameters: dict = {phase: copy.deepcopy(hyper_parameter)}
        if not hook_config:
            hook_config = HookConfig()
        self.hook_config: HookConfig = copy.deepcopy(hook_config)
        self.__device: None | torch.device = None
        self.__device_fun: Callable = DeviceGreedyAllocator.get_device
        self.__dataloader: None | torch.utils.data.DataLoader = None
        self.__dataloader_kwargs: dict = (
            copy.deepcopy(dataloader_kwargs) if dataloader_kwargs is not None else {}
        )
        self.__stream: None | torch.cpu.Stream | torch.Stream = None
        self.__save_dir: None | str = None
        self.__visualizer_prefix: str = ""
        self.set_default_device: bool = False

    @property
    def mutable_model_config(self) -> ModelConfig:
        assert self.__model_config is not None
        assert self.__model_evaluator is None
        return self.__model_config

    @property
    def dataset_collection_config(self) -> DatasetCollectionConfig:
        return self.__dataset_collection_config

    @property
    def dataset_collection(self) -> DatasetCollection:
        if self.__dataset_collection is None:
            self.__dataset_collection = (
                self.__dataset_collection_config.create_dataset_collection(
                    save_dir=self.save_dir
                )
            )
        return self.__dataset_collection

    def remove_unrelated_datasets(self) -> None:
        if self.__dataset_collection is None:
            self.__dataset_collection_config.keep_phases = {self.phase}
        else:
            for phase in list(self.dataset_collection.foreach_phase()):
                if phase != self.phase:
                    self.dataset_collection.remove_dataset(phase=phase)

    @property
    def mutable_dataset_collection(self) -> DatasetCollection:
        self.__dataloader = None
        return self.dataset_collection

    def has_device(self) -> bool:
        return self.__device is not None

    @property
    def device(self) -> torch.device:
        if self.__device is None:
            self.set_device(self.__device_fun())
        assert self.__device is not None
        return self.__device

    @property
    def stream(self) -> torch.cpu.Stream | torch.Stream:
        if self.__stream is None:
            match self.device.type.lower():
                case "cuda":
                    self.__stream = torch.cuda.Stream(device=self.device)
                case "cpu" | "mps":
                    self.__stream = torch.cpu.Stream()
                case _:
                    raise RuntimeError(self.device)
        assert self.__stream is not None
        return self.__stream

    @property
    def stream_context(
        self,
    ) -> AbstractContextManager:
        s = self.stream
        match self.device.type.lower():
            case "cuda":
                assert isinstance(s, torch.cuda.Stream)
                return torch.cuda.stream(s)
            case "cpu" | "mps":
                assert isinstance(s, torch.cpu.Stream)
                return torch.cpu.stream(s)
        raise RuntimeError(self.device)

    @property
    def complete_stream_context(self) -> AbstractContextManager:
        stack = ExitStack()
        stack.enter_context(SyncedStreamContext(self.stream))
        stack.enter_context(self.stream_context)
        if self.set_default_device:
            stack.enter_context(DefaultDeviceContext(self.device))
        return stack

    @property
    def dataloader_kwargs(self) -> dict:
        return self.__dataloader_kwargs

    @property
    def hyper_parameter(self) -> HyperParameter:
        return self.__hyper_parameters[self.phase]

    def set_phase(self, phase: MachineLearningPhase) -> None:
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
        for hook in self._hook_objs.values():
            if isinstance(hook, MetricVisualizer):
                hook.set_data_dir(data_dir)
        for executor in self._foreach_sub_executor():
            executor.set_save_dir(save_dir)

    def set_visualizer_prefix(self, prefix: str) -> None:
        self.__visualizer_prefix = prefix
        for hook in self._hook_objs.values():
            if isinstance(hook, MetricVisualizer):
                hook.set_prefix(prefix)

    @property
    def visualizer_prefix(self) -> str:
        return self.__visualizer_prefix

    @property
    def save_dir(self) -> None | str:
        return self.__save_dir

    @functools.cached_property
    def dataset_util(self) -> DatasetUtil:
        return self.dataset_collection.get_dataset_util(phase=self.__phase)

    @property
    def dataset_size(self) -> int:
        return len(self.dataset_util)

    def remove_dataloader_kwargs(self, key: str) -> None:
        self.__dataloader_kwargs.pop(key, None)
        self.__dataloader = None

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
                **self.__dataloader_kwargs,
            )
        return self.__dataloader

    @property
    def running_model_evaluator(self) -> ModelEvaluator:
        if self.__model_evaluator is None and self.__auto_create_model:
            self.recover_model()
        assert self.__model_evaluator is not None
        return self.__model_evaluator

    @property
    def model_evaluator(self) -> ModelEvaluator:
        self.wait_stream()
        return self.running_model_evaluator

    @property
    def model_util(self) -> ModelUtil:
        return self.running_model_evaluator.model_util

    @property
    def model(self) -> torch.nn.Module:
        return self.running_model_evaluator.model

    def remove_model(self) -> None:
        self.__model_evaluator = None

    def recover_model(self) -> None:
        if self.__model_evaluator is None:
            assert self.__model_config is not None
            self.__model_evaluator = self.__model_config.get_model(
                dc=self.dataset_collection
            )

    def replace_model(self, fun: Callable) -> None:
        self.running_model_evaluator.set_model(fun(self.model))

    def replace_model_evaluator(self, fun: Callable) -> None:
        self.wait_stream()
        self.__model_evaluator = fun(self.model_evaluator)

    def _prepare_execution(self) -> None:
        self.hook_config.set_hooks(self)
        if self.save_dir:
            self.set_save_dir(self.save_dir)
        if self.__visualizer_prefix:
            self.set_visualizer_prefix(self.__visualizer_prefix)
        self.exec_hooks(hook_point=ExecutorHookPoint.BEFORE_EXECUTE)

    def set_device_fun(self, device_fun: Callable) -> None:
        self.__device_fun = device_fun

    def set_device(self, device: torch.device) -> None:
        if self.__device != device:
            self.wait_stream()
            self.__device = device
            log_debug("%s use device %s", str(self.__phase), self.__device)
            self.__stream = None
            self.__dataloader = None

        for executor in self._foreach_sub_executor():
            executor.set_device(device)

    def __getstate__(self) -> dict[str, Any]:
        # capture what is normally pickled
        state = self.__dict__.copy()
        state["_Executor__device"] = None
        state["_Executor__stream"] = None
        state["_Executor__dataloader"] = None
        return state

    def wait_stream(self) -> None:
        if self.__stream is not None:
            with SyncedStreamContext(self.__stream):
                pass

    def set_dataset_collection(self, dc: DatasetCollection) -> None:
        self.wait_stream()
        self.__dataset_collection = dc

    def set_model_evaluator(self, model_evaluator: ModelEvaluator) -> None:
        self.wait_stream()
        self.__model_evaluator = model_evaluator

    def _foreach_sub_executor(self) -> Generator:
        yield from []

    def save_model(self, model_path: str) -> None:
        torch.save(self.model.state_dict(), model_path)

    def offload_from_device(self) -> None:
        self.wait_stream()
        if self.__model_evaluator:
            self.model_evaluator.offload_from_device()
        for executor in self._foreach_sub_executor():
            executor.offload_from_device()
        gc.collect()
        torch.cuda.empty_cache()

    def has_optimizer(self) -> bool:
        return "optimizer" in self._data

    def has_lr_scheduler(self) -> bool:
        return "lr_scheduler" in self._data

    def get_optimizer(self) -> torch.optim.Optimizer:
        raise NotImplementedError()

    def get_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        if "lr_scheduler" not in self._data:
            self._data["lr_scheduler"] = self.hyper_parameter.get_lr_scheduler(self)
        return self._data["lr_scheduler"]

    def __execute_batch(
        self,
        batch_index: int,
        batch: dict,
        epoch: int,
        evaluation_mode: EvaluationMode,
    ) -> None:
        if (
            evaluation_mode == EvaluationMode.Training
            and self.hyper_parameter.batch_size != 1
            and batch.get("batch_size") == 1
            and self.running_model_evaluator.model_util.have_module(
                module_type=torch.nn.BatchNorm2d
            )
        ):
            log_debug("drop last one-sized batch for batch norm")
            return
        batch |= {
            "batch_index": batch_index,
            "phase": self.phase,
            "device": self.device,
            "evaluation_mode": evaluation_mode,
            "non_blocking": True,
        }

        self.exec_hooks(
            hook_point=ExecutorHookPoint.BEFORE_BATCH,
            epoch=epoch,
            **batch,
        )

        evaluation_kwargs = batch
        forward_result: dict = {}
        if self.has_hook(ExecutorHookPoint.MODEL_FORWARD):
            self.exec_hooks(
                hook_point=ExecutorHookPoint.MODEL_FORWARD,
                evaluation_kwargs=evaluation_kwargs,
            )
            forward_result = self._data.pop("forward_result")
        else:
            forward_result = self.running_model_evaluator(**evaluation_kwargs)

        batch |= forward_result
        if evaluation_mode in (EvaluationMode.Training, EvaluationMode.TestWithGrad):
            if evaluation_mode == EvaluationMode.Training:
                optimizer = self.get_optimizer()
                self.running_model_evaluator.backward_and_step(
                    loss=forward_result["loss"], optimizer=optimizer
                )
            else:
                normalized_batch_loss = (
                    self.running_model_evaluator.get_normalized_batch_loss(
                        dataset_util=self.dataset_util,
                        forward_result=forward_result,
                    )
                )
                self.running_model_evaluator.backward(loss=normalized_batch_loss)

            if evaluation_mode == EvaluationMode.Training:
                lr_scheduler = self.get_lr_scheduler()
                if lr_scheduler_step_after_batch(lr_scheduler):
                    log_debug("adjust lr after batch")
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
        self.exec_hooks(hook_point=ExecutorHookPoint.BEFORE_FETCH_BATCH, batch_index=0)
        for batch_index, batch in enumerate(self.dataloader):
            self.exec_hooks(
                hook_point=ExecutorHookPoint.AFTER_FETCH_BATCH,
                batch_index=batch_index,
            )
            self.__execute_batch(
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
            if not lr_scheduler_step_after_batch(lr_scheduler) and not isinstance(
                lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                lr_scheduler.step()

        self.exec_hooks(
            hook_point=ExecutorHookPoint.AFTER_EPOCH,
            epoch=epoch,
        )


@dataclass(kw_only=True)
class ExecutorConfig(ConfigBase):
    hook_config: HookConfig = field(default_factory=HookConfig)
    dataloader_kwargs: dict = field(default_factory=dict)
    cache_transforms: None | str = None

    def create_executor(
        self,
        cls: Callable,
        **kwargs,
    ) -> Any:
        if (
            self.cache_transforms is not None
            and "cache_transforms" not in self.dataloader_kwargs
        ):
            self.dataloader_kwargs["cache_transforms"] = self.cache_transforms
        return cls(
            hook_config=self.hook_config,
            dataloader_kwargs=self.dataloader_kwargs,
            **kwargs,
        )
