import abc
import contextlib
import copy
import os
from typing import Any, Callable

import torch
from cyy_naive_lib.log import get_logger

from .dataloader import get_dataloader
from .dataset.util import DatasetUtil
from .dataset_collection import DatasetCollection
from .device import get_device
from .hook import HookCollection
from .hook.config import HookConfig
from .hook.executor_logger import ExecutorLogger
from .hyper_parameter import HyperParameter
from .metric_visualizers.metric_visualizer import MetricVisualizer
from .metrics.performance_metric import PerformanceMetric
from .ml_type import ExecutorHookPoint, MachineLearningPhase
from .model_evaluator import ModelEvaluator
from .model_util import ModelUtil

# from cyy_torch_toolbox.metric_visualizers.metric_tensorboard import \
#     MetricTensorBoard


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
        self.__dataloader = None
        self.__dataloader_kwargs: dict = {}
        self.__device_stream: None | torch.cuda.Stream = None
        self.append_hook(ExecutorLogger(), "logger")
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
        for hook in self.get_hooks():
            if isinstance(hook, MetricVisualizer):
                hook.set_data_dir(data_dir)

    def set_visualizer_prefix(self, prefix: str) -> None:
        for hook in self.get_hooks():
            if isinstance(hook, MetricVisualizer):
                hook.set_prefix(prefix)

    @property
    def visualizer_prefix(self) -> None | str:
        for hook in self.get_hooks():
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

    def transform_dataset(self, transformer: Callable) -> None:
        self.dataset_collection.transform_dataset(self.phase, transformer)

    @property
    def dataset_size(self) -> int:
        if "dataset_size" not in self._data:
            self.refresh_dataset_size()
        return self._data["dataset_size"]

    def refresh_dataset_size(self) -> None:
        self._data["dataset_size"] = len(self.dataset_util)

    def update_dataloader_kwargs(self, **kwargs: Any) -> None:
        self.__dataloader_kwargs.update(kwargs)
        self.__dataloader = None

    @property
    def dataloader(self):
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

    def replace_model_evaluator(self, fun: Callable) -> None:
        self.__model_evaluator = fun(self.model_evaluator)

    def replace_model(self, fun: Callable) -> None:
        self.__model_evaluator = self.model_evaluator.replace_model(fun(self.model))

    def _prepare_execution(self, **kwargs: Any) -> None:
        self._data.clear()
        for k, v in kwargs.items():
            if not hasattr(self.hook_config, k):
                continue
            setattr(self.hook_config, k, v)
        self.hook_config.set_hooks(self)
        if self.__save_dir is not None:
            self.set_save_dir(self.__save_dir)

        self.exec_hooks(hook_point=ExecutorHookPoint.BEFORE_EXECUTE)

    def set_device(self, device: torch.device) -> None:
        if self.__device == device:
            return
        self.wait_stream()
        self.__device = device
        get_logger().debug("%s use device %s", str(self.__phase), self.__device)
        self.__device_stream = None
        self.__dataloader = None

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
            if "cuda" not in self.device.type.lower()
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

    def save_model(self, model_path: str) -> None:
        torch.save(self.model.state_dict(), model_path)

    def offload_from_device(self) -> None:
        self.wait_stream()
        self.__model_evaluator.offload_from_device()
        torch.cuda.empty_cache()

    def has_optimizer(self) -> bool:
        return "optimizer" in self._data

    def has_lr_scheduler(self) -> bool:
        return "lr_scheduler" in self._data

    def get_optimizer(self) -> torch.optim.Optimizer:
        if "optimizer" not in self._data:
            self._data["optimizer"] = self.hyper_parameter.get_optimizer(self)
        return self._data["optimizer"]

    def get_lr_scheduler(self) -> Any:
        if "lr_scheduler" not in self._data:
            self._data["lr_scheduler"] = self.hyper_parameter.get_lr_scheduler(self)
        return self._data["lr_scheduler"]

    def _execute_epoch(
        self,
        epoch: int,
        in_training: bool,
        need_backward: bool = False,
        reduce_loss: bool = True,
    ) -> None:
        step_lr_after_epoch: bool = False
        self.exec_hooks(
            hook_point=ExecutorHookPoint.BEFORE_EPOCH,
            epoch=epoch,
        )
        self.refresh_dataset_size()
        self.exec_hooks(hook_point=ExecutorHookPoint.BEFORE_FETCH_BATCH, batch_index=0)
        for batch_index, batch in enumerate(self.dataloader):
            self.exec_hooks(
                hook_point=ExecutorHookPoint.AFTER_FETCH_BATCH,
                batch_index=batch_index,
            )
            batch["batch_index"] = batch_index
            if in_training:
                if (
                    self.hyper_parameter.batch_size != 1
                    and batch.get("batch_size", None) == 1
                    and self.__model_evaluator.model_util.have_module(
                        module_type=torch.nn.BatchNorm2d
                    )
                ):
                    get_logger().debug("drop last one-batch for batchnorm")
                    continue
                need_backward = True
                optimizer: torch.optim.Optimizer = self.get_optimizer()
                lr_scheduler = self.get_lr_scheduler()

            self.exec_hooks(
                hook_point=ExecutorHookPoint.BEFORE_BATCH,
                epoch=epoch,
                **batch,
            )
            kwargs = batch | {
                "phase": self.phase,
                "device": self.device,
                "need_backward": need_backward,
                "non_blocking": True,
                "reduce_loss": reduce_loss,
            }

            forward_result: dict = {}

            while True:
                with torch.set_grad_enabled(mode=need_backward):
                    if need_backward:
                        if in_training:
                            optimizer.zero_grad(set_to_none=True)
                        else:
                            self.running_model_evaluator.model.zero_grad(
                                set_to_none=True
                            )
                    if self.has_hook(ExecutorHookPoint.MODEL_FORWARD):
                        self.exec_hooks(
                            hook_point=ExecutorHookPoint.MODEL_FORWARD,
                            model_kwargs=kwargs,
                        )
                        forward_result = self._data.pop("forward_result")
                    else:
                        forward_result = self.__model_evaluator(**kwargs)

                if forward_result["is_averaged_loss"]:
                    assert self._data["dataset_size"] > 1
                    forward_result["normalized_batch_loss"] = (
                        forward_result["loss"]
                        * forward_result["loss_batch_size"]
                        / self._data["dataset_size"]
                    )
                else:
                    forward_result["normalized_batch_loss"] = forward_result["loss"]
                batch |= forward_result
                self.exec_hooks(
                    hook_point=ExecutorHookPoint.AFTER_FORWARD,
                    epoch=epoch,
                    **batch,
                )

                if need_backward:
                    loss = self._get_backward_loss(result=forward_result)
                    assert loss is not None
                    if self.has_hook(ExecutorHookPoint.MODEL_BACKWARD):
                        self.exec_hooks(
                            hook_point=ExecutorHookPoint.MODEL_BACKWARD, loss=loss
                        )
                    else:
                        get_logger().error("loss is %s", loss)
                        loss.backward()

                if not in_training:
                    break
                step_skipped: bool = False
                if self.has_hook(ExecutorHookPoint.OPTIMIZER_STEP):
                    self._data["step_skipped"] = False
                    self.exec_hooks(ExecutorHookPoint.OPTIMIZER_STEP)
                    step_skipped = self._data["step_skipped"]
                else:
                    optimizer.step()
                if step_skipped:
                    self.exec_hooks(
                        hook_point=ExecutorHookPoint.CANCEL_FORWARD,
                        epoch=epoch,
                        **batch,
                    )
                    continue
                if HyperParameter.lr_scheduler_step_after_batch(lr_scheduler):
                    get_logger().debug("adjust lr after batch")
                    lr_scheduler.step()
                    step_lr_after_epoch = False
                else:
                    step_lr_after_epoch = True
                break

            self.exec_hooks(
                hook_point=ExecutorHookPoint.AFTER_BATCH,
                epoch=epoch,
                result=forward_result,
                **batch,
            )

            self.exec_hooks(
                hook_point=ExecutorHookPoint.BEFORE_FETCH_BATCH,
                batch_index=batch_index + 1,
            )
        if in_training and step_lr_after_epoch:
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
