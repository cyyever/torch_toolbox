import copy
import os
import shutil
from typing import Callable

import torch
from cyy_naive_lib.log import get_logger

from cyy_torch_toolbox.dataloader.dataloader import get_dataloader
from cyy_torch_toolbox.dataset_collection import DatasetCollection
from cyy_torch_toolbox.dataset_util import DatasetUtil
from cyy_torch_toolbox.device import get_device
from cyy_torch_toolbox.hook_config import HookConfig
from cyy_torch_toolbox.hooks.model_executor_logger import ModelExecutorLogger
from cyy_torch_toolbox.hyper_parameter import HyperParameter
# from cyy_torch_toolbox.metric_visualizers.metric_tensorboard import \
#     MetricTensorBoard
from cyy_torch_toolbox.metric_visualizers.metric_visualizer import \
    MetricVisualizer
from cyy_torch_toolbox.metric_visualizers.performance_metric_logger import \
    PerformanceMetricLogger
from cyy_torch_toolbox.metrics.performance_metric import PerformanceMetric
from cyy_torch_toolbox.ml_type import (DatasetType, MachineLearningPhase,
                                       ModelExecutorHookPoint)
from cyy_torch_toolbox.model_executor_base import ModelExecutorBase
from cyy_torch_toolbox.model_util import ModelUtil
from cyy_torch_toolbox.model_with_loss import ModelWithLoss


class ModelExecutor(ModelExecutorBase):
    def __init__(
        self,
        model_with_loss: ModelWithLoss,
        dataset_collection: DatasetCollection,
        phase: MachineLearningPhase,
        hyper_parameter: HyperParameter,
        hook_config: HookConfig | None = None,
    ) -> None:
        super().__init__()
        self._model_with_loss = model_with_loss
        self.__dataset_collection: DatasetCollection = dataset_collection
        self.__phase = phase
        self.__hyper_parameter = hyper_parameter
        self._hook_config = hook_config
        self.__device = None
        self.__dataloader = None
        self.__cuda_stream = None
        self.append_hook(ModelExecutorLogger(), "logger")
        self.append_hook(
            PerformanceMetric(
                model_type=self._model_with_loss.model_type,
                profile=hook_config.profile if hook_config is not None else False,
            ),
            "performance_metric",
        )
        self.append_hook(PerformanceMetricLogger(), "performance_metric_logger")
        # self.append_hook(MetricTensorBoard(), "tensor_board_visualizer")
        self.__save_dir: None | str = None
        self._visualizer_prefix: None | str = None
        self.cache_transforms = None

    @property
    def hyper_parameter(self):
        return self.__hyper_parameter

    def _get_batch_size(self) -> int:
        return self.hyper_parameter.batch_size

    @property
    def performance_metric(self):
        return self.get_hook("performance_metric")

    @property
    def phase(self):
        return self.__phase

    def set_save_dir(self, save_dir: str) -> None:
        self.__save_dir = save_dir
        if save_dir is not None:
            data_dir = os.path.join(save_dir, "visualizer")
            os.makedirs(data_dir, exist_ok=True)
            for hook in self.get_hooks():
                if isinstance(hook, MetricVisualizer):
                    hook.set_data_dir(data_dir)

    def set_visualizer_prefix(self, prefix: str) -> None:
        self._visualizer_prefix = prefix
        for hook in self.get_hooks():
            if isinstance(hook, MetricVisualizer):
                hook.set_prefix(prefix)

    @property
    def save_dir(self):
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
        return len(self.dataset_util)

    @property
    def dataloader(self):
        if self.__dataloader is None:
            self.__dataloader = get_dataloader(
                dc=self.dataset_collection,
                phase=self.__phase,
                batch_size=self._get_batch_size(),
                device=self.device,
                model_type=self._model_with_loss.model_type,
                cache_transforms=self.cache_transforms,
            )
        return self.__dataloader

    @property
    def model_with_loss(self) -> ModelWithLoss:
        self._wait_stream()
        return self._model_with_loss

    @property
    def model_util(self) -> ModelUtil:
        return self.model_with_loss.model_util

    @property
    def loss_fun(self):
        return self._model_with_loss.loss_fun

    @property
    def model(self) -> torch.nn.Module:
        return self.model_with_loss.model

    def replace_model(self, fun: Callable) -> None:
        self._model_with_loss = self.model_with_loss.replace_model(fun(self.model))

    def copy_model_with_loss(self, deepcopy: bool = True) -> ModelWithLoss:
        self._wait_stream()
        if deepcopy:
            return copy.deepcopy(self._model_with_loss)
        return copy.copy(self._model_with_loss)

    def _prepare_execution(self, **kwargs):
        self._data.clear()
        if self._hook_config:
            for k, v in kwargs.items():
                if not hasattr(self._hook_config, k):
                    continue
                setattr(self._hook_config, k, v)
            self._hook_config.append_hooks(self)
        if self.__save_dir is not None:
            self.set_save_dir(self.__save_dir)

        if self._visualizer_prefix is not None:
            self.set_visualizer_prefix(self._visualizer_prefix)
        self._data["dataset_size"] = self.dataset_size
        self.exec_hooks(ModelExecutorHookPoint.BEFORE_EXECUTE)

    @property
    def dataset_collection(self) -> DatasetCollection:
        return self.__dataset_collection

    @property
    def device(self):
        if self.__device is None:
            self.set_device(get_device())
        return self.__device

    def set_device(self, device):
        if self.__device == device:
            return
        self._wait_stream()
        self.__device = device
        get_logger().info("%s use device %s", str(self.__phase), self.__device)
        self.__cuda_stream = None
        self.__dataloader = None

    def __getstate__(self):
        # capture what is normally pickled
        state = self.__dict__.copy()
        state["_ModelExecutor__cuda_stream"] = None
        state["_ModelExecutor__dataloader"] = None
        return state

    @property
    def cuda_stream(self):
        if self.__cuda_stream is None and "cuda" in self.device.type.lower():
            self.__cuda_stream = torch.cuda.Stream(device=self.device)
            self.__cuda_stream.wait_stream(torch.cuda.current_stream())
        return self.__cuda_stream

    def wait_stream(self):
        self._wait_stream()

    def _wait_stream(self):
        if self.__cuda_stream is not None:
            self.__cuda_stream.synchronize()
            assert self.__cuda_stream.query()

    def set_dataset_collection(self, dc: DatasetCollection) -> None:
        self._wait_stream()
        self.__dataset_collection = dc
        if self.save_dir is not None:
            shutil.rmtree(os.path.join(self.save_dir, "dc.pk"), ignore_errors=True)

    def set_model_with_loss(self, model_with_loss: ModelWithLoss) -> None:
        self._wait_stream()
        self._model_with_loss = model_with_loss
        if self.save_dir is not None:
            shutil.rmtree(
                os.path.join(self.save_dir, "model_and_loss.pk"), ignore_errors=True
            )

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def offload_from_gpu(self):
        self._wait_stream()
        self._model_with_loss.offload_from_gpu()
        # if self.__dataloader is not None:
        #     del self.__dataloader
        #     self.__dataloader = None
        torch.cuda.empty_cache()

    def split_batch_input(self, inputs, targets, input_features=None):
        batch_dim = 0
        if self.dataset_collection.dataset_type == DatasetType.Text:
            if "BatchEncoding" in type(inputs).__name__:
                new_inputs = []
                first_value = next(iter(inputs.values()))
                assert isinstance(first_value, torch.Tensor)
                for i in range(first_value.size(dim=0)):
                    new_inputs.append(
                        {k: v[i].unsqueeze(dim=0) for k, v in inputs.items()}
                    )
                inputs = new_inputs

            if isinstance(inputs, torch.Tensor):
                if (
                    batch_dim == 0
                    and inputs.shape[0] != targets.shape[0]
                    and inputs.shape[1] == targets.shape[0]
                ):
                    batch_dim = 1
                if batch_dim != 0:
                    inputs = inputs.permute(batch_dim, 0)
            if batch_dim != 0 and isinstance(input_features, torch.Tensor):
                input_features = input_features.permute(batch_dim, 0, 2)
        return inputs, batch_dim, input_features

    def get_optimizer(self):
        return None

    def get_lr_scheduler(self):
        return None

    def _execute_epoch(
        self, epoch: int, in_training: bool, need_backward: bool = False
    ) -> None:
        self._data["_all_steps_skipped_in_epoch"] = True
        self.exec_hooks(
            hook_point=ModelExecutorHookPoint.BEFORE_EPOCH,
            epoch=epoch,
        )
        self.exec_hooks(ModelExecutorHookPoint.BEFORE_FETCH_BATCH, batch_index=0)
        for batch_index, batch in enumerate(self.dataloader):
            self.exec_hooks(
                ModelExecutorHookPoint.AFTER_FETCH_BATCH,
                batch_index=batch_index,
            )
            batch["batch_index"] = batch_index
            if in_training:
                if (
                    self._get_batch_size() != 1
                    and batch["batch_size"] == 1
                    and self._model_with_loss.model_util.have_module(
                        module_type=torch.nn.BatchNorm2d
                    )
                ):
                    get_logger().debug("drop last one-batch for batchnorm")
                    continue
                need_backward = True
                optimizer = self.get_optimizer()
                optimizer.zero_grad(set_to_none=True)

            self.exec_hooks(
                ModelExecutorHookPoint.BEFORE_BATCH,
                epoch=epoch,
                **batch,
            )
            kwargs = batch | {
                "phase": self.phase,
                "device": self.device,
                "need_backward": need_backward,
                "non_blocking": True,
            }
            if self.has_hook(ModelExecutorHookPoint.MODEL_FORWARD):
                self.exec_hooks(
                    ModelExecutorHookPoint.MODEL_FORWARD,
                    model_kwargs=kwargs,
                )
                result = self._data.pop("forward_result")
            else:
                result = self._model_with_loss(**kwargs)

            if result["is_averaged_loss"]:
                normalized_batch_loss = (
                    result["loss"] * batch["batch_size"] / self._data["dataset_size"]
                )
            else:
                assert False
                normalized_batch_loss = result["loss"] / self._data["dataset_size"]
            result["normalized_batch_loss"] = normalized_batch_loss
            batch["cpu_inputs"] = result["cpu_inputs"]
            batch["inputs"] = result["inputs"]
            batch["targets"] = result["targets"]
            batch["input_features"] = result["input_features"]
            self.exec_hooks(
                ModelExecutorHookPoint.AFTER_FORWARD,
                epoch=epoch,
                **batch,
            )

            if need_backward:
                loss = self._get_backward_loss(result=result)
                assert loss is not None
                if self.has_hook(ModelExecutorHookPoint.MODEL_BACKWARD):
                    self.exec_hooks(ModelExecutorHookPoint.MODEL_BACKWARD, loss=loss)
                else:
                    loss.backward()

            self.exec_hooks(
                ModelExecutorHookPoint.AFTER_BATCH,
                epoch=epoch,
                result=result,
                **batch,
            )
            if in_training:
                step_skipped: bool = False
                if self.has_hook(ModelExecutorHookPoint.OPTIMIZER_STEP):
                    self._data["step_skipped"] = False
                    self.exec_hooks(ModelExecutorHookPoint.OPTIMIZER_STEP)
                    step_skipped = self._data["step_skipped"]
                else:
                    optimizer.step()
                if not step_skipped:
                    self._data["_all_steps_skipped_in_epoch"] = False
                    lr_scheduler = self.get_lr_scheduler()
                    if HyperParameter.lr_scheduler_step_after_batch(lr_scheduler):
                        get_logger().debug("adjust lr after batch")
                        lr_scheduler.step()

                self.exec_hooks(
                    ModelExecutorHookPoint.AFTER_OPTIMIZER_STEP,
                    epoch=epoch,
                    step_skipped=step_skipped,
                    **batch,
                )

            self.exec_hooks(
                ModelExecutorHookPoint.BEFORE_FETCH_BATCH,
                batch_index=batch_index + 1,
            )

        self.exec_hooks(
            ModelExecutorHookPoint.AFTER_EPOCH,
            epoch=epoch,
        )

    def _get_backward_loss(self, result):
        raise NotImplementedError()
