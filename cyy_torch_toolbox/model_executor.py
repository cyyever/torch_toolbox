import copy
import os
import pickle
import shutil
from typing import Callable, Optional

import torch
from cyy_naive_lib.log import get_logger

from cyy_torch_toolbox.dataloader import get_dataloader
from cyy_torch_toolbox.dataset import get_dataset_size
from cyy_torch_toolbox.dataset_collection import DatasetCollection
from cyy_torch_toolbox.dataset_util import DatasetUtil
from cyy_torch_toolbox.device import get_device
from cyy_torch_toolbox.hooks.amp import AMP
from cyy_torch_toolbox.hooks.model_executor_logger import ModelExecutorLogger
from cyy_torch_toolbox.hooks.profiler import Profiler
from cyy_torch_toolbox.hyper_parameter import HyperParameter
from cyy_torch_toolbox.metric_visualizers.metric_tensorboard import \
    MetricTensorBoard
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
    ):
        super().__init__()
        self._model_with_loss = model_with_loss
        self.__dataset_collection: DatasetCollection = dataset_collection
        self.__phase = phase
        self.__hyper_parameter = hyper_parameter
        self.__device = None
        self.__dataloader = None
        self.__cuda_stream = None
        self.__logger = ModelExecutorLogger()
        self.append_hook(self.__logger)
        self.__profiler = Profiler()
        self.append_hook(self.__profiler)
        self.__performance_metric = PerformanceMetric(self._model_with_loss.model_type)
        self.append_hook(self.__performance_metric)
        self.__performance_metric_logger = PerformanceMetricLogger()
        self.append_hook(self.__performance_metric_logger)
        self.__metric_tb: MetricTensorBoard = MetricTensorBoard()
        self.append_hook(self.__metric_tb)
        self.debugging_mode = False
        self.profiling_mode = False
        self.__save_dir: Optional[str] = None
        self.cache_transforms = None
        self.__amp_hook = None

    @property
    def visualizer(self):
        return self.__metric_tb

    @property
    def phase(self):
        return self.__phase

    @property
    def performance_metric(self):
        return self.__performance_metric

    @property
    def performance_metric_logger(self):
        return self.__performance_metric_logger

    def set_save_dir(self, save_dir: str) -> None:
        self.__save_dir = save_dir
        if save_dir is not None:
            log_dir = os.path.join(save_dir, "visualizer")
            os.makedirs(log_dir, exist_ok=True)
            self.__metric_tb.set_log_dir(log_dir)

    @property
    def save_dir(self):
        return self.__save_dir

    def disable_logger(self):
        self.disable_hook(self.__logger)

    def disable_performance_metric_logger(self):
        self.disable_hook(self.__performance_metric_logger)

    @property
    def dataset(self):
        return self.dataset_collection.get_dataset(phase=self.__phase)

    @property
    def dataset_util(self) -> DatasetUtil:
        return self.dataset_collection.get_dataset_util(phase=self.__phase)

    @property
    def dataset_size(self):
        return get_dataset_size(self.dataset)

    def transform_dataset(self, transformer: Callable) -> None:
        self.dataset_collection.transform_dataset(self.phase, transformer)

    @property
    def dataloader(self):
        if self.__dataloader is None:
            self.__dataloader = get_dataloader(
                dc=self.dataset_collection,
                model_type=self._model_with_loss.model_type,
                phase=self.__phase,
                hyper_parameter=self.__hyper_parameter,
                device=self.device,
                cache_transforms=self.cache_transforms,
            )
            self.cache_transforms = None
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

    def copy_model_with_loss(self, deepcopy=True):
        self._wait_stream()
        if deepcopy:
            return copy.deepcopy(self._model_with_loss)
        return copy.copy(self._model_with_loss)

    def has_amp(self) -> bool:
        return self.__amp_hook is not None

    def set_amp(self, enabled=True):
        if self.__amp_hook is not None:
            self.remove_hook(self.__amp_hook)
            self.__amp_hook = None
        if enabled:
            self.__amp_hook = AMP()
            self.append_hook(self.__amp_hook)
        get_logger().debug("use AMP")

    def _prepare_execution(self, **kwargs):
        self.clear_data()
        for name in dir(self):
            if isinstance(getattr(type(self), name, None), property):
                continue
            attr = getattr(self, name)
            if hasattr(attr, "_is_cyy_torch_toolbox_metric"):
                attr.clear_metric()

        if self.profiling_mode:
            get_logger().warning("use profiling mode")
            self.enable_hook(self.__profiler)
        else:
            self.disable_hook(self.__profiler)

    @property
    def dataset_collection(self) -> DatasetCollection:
        return self.__dataset_collection

    @property
    def device(self):
        if self.__device is None:
            self.set_device(get_device())
        return self.__device

    def set_device(self, device):
        self._wait_stream()
        self.__device = device
        get_logger().debug("%s use device %s", str(self.__phase), self.__device)
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

    def _wait_stream(self):
        if self.__cuda_stream is not None:
            self.__cuda_stream.synchronize()
            if self.debugging_mode:
                assert self.__cuda_stream.query()

    def set_hyper_parameter(self, hyper_parameter):
        self.__hyper_parameter = hyper_parameter
        self.__dataloader = None

    @property
    def hyper_parameter(self):
        return self.__hyper_parameter

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

    def get_batch_size(self, batch):
        match batch:
            case tuple():
                return self.get_batch_size(batch[1])
            case torch.Tensor():
                return batch.shape[0]
        raise RuntimeError("invalid batch:" + str(batch))

    def offload_from_gpu(self):
        self._wait_stream()
        self._model_with_loss.offload_from_gpu()
        if self.__dataloader is not None:
            del self.__dataloader
            self.__dataloader = None
        torch.cuda.empty_cache()

    def offload_from_memory(self):
        assert self.save_dir is not None
        self.offload_from_gpu()
        with open(os.path.join(self.save_dir, "model_and_loss.pk"), "wb") as file:
            pickle.dump(
                self.model_with_loss,
                file,
            )
            self._model_with_loss = None
        with open(os.path.join(self.save_dir, "dc.pk"), "wb") as file:
            pickle.dump(
                self.__dataset_collection,
                file,
            )
            self.__dataset_collection = None

    def load_to_memory(self):
        assert self.save_dir is not None
        if self._model_with_loss is None:
            with open(os.path.join(self.save_dir, "model_and_loss.pk"), "rb") as file:
                self._model_with_loss = pickle.load(file)
        if self.__dataset_collection is None:
            with open(os.path.join(self.save_dir, "dc.pk"), "rb") as file:
                self.__dataset_collection = pickle.load(file)

    def split_batch_input(self, inputs, targets):
        batch_dim = 0
        if self.dataset_collection.dataset_type == DatasetType.Text:
            if "BatchEncoding" in type(inputs).__name__:
                new_inputs = []
                first_value = next(iter(inputs.values()))
                assert isinstance(first_value, torch.Tensor)
                for i in range(first_value.size(dim=0)):
                    new_inputs.append({k: v[i] for k, v in inputs.items()})
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
        return inputs, batch_dim

    def get_optimizer(self):
        raise NotImplementedError()

    def get_lr_scheduler(self):
        raise NotImplementedError()

    def _execute_epoch(
        self, epoch: int, need_backward: bool, in_training: bool
    ) -> None:
        if epoch in self.get_data("skipped_epoch", set()):
            get_logger().warning("skip epoch %s", epoch)
            return
        self.exec_hooks(
            ModelExecutorHookPoint.BEFORE_EPOCH,
            epoch=epoch,
        )
        self.exec_hooks(ModelExecutorHookPoint.BEFORE_FETCH_BATCH, batch_index=0)
        for batch_index, batch in enumerate(self.dataloader):
            self.exec_hooks(
                ModelExecutorHookPoint.AFTER_FETCH_BATCH,
                batch_index=batch_index,
            )
            if in_training:
                optimizer = self.get_optimizer()
                optimizer.zero_grad(set_to_none=True)

            if (
                in_training
                and self.hyper_parameter.batch_size != 1
                and batch["batch_size"] == 1
                and self._model_with_loss.model_util.have_module(
                    module_type=torch.nn.BatchNorm2d
                )
            ):
                get_logger().debug("drop last one-batch for batchnorm")
                continue

            self.exec_hooks(
                ModelExecutorHookPoint.BEFORE_BATCH,
                batch_index=batch_index,
                **batch,
            )
            kwargs = {
                "inputs": batch["inputs"],
                "targets": batch["targets"],
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
                result = self.pop_data("forward_result")
            else:
                result = self._model_with_loss(**kwargs)

            if result["is_averaged_loss"]:
                normalized_batch_loss = (
                    result["loss"] * batch["batch_size"] / self.dataset_size
                )
            else:
                normalized_batch_loss = result["loss"] / self.dataset_size
            result["normalized_batch_loss"] = normalized_batch_loss
            batch["cpu_inputs"] = batch["inputs"]
            batch["cpu_targets"] = batch["targets"]
            batch["inputs"] = result["inputs"]
            batch["targets"] = result["targets"]
            self.exec_hooks(
                ModelExecutorHookPoint.AFTER_FORWARD,
                input_features=result["input_features"],
                epoch=epoch,
                **batch,
            )

            loss = self._get_backward_loss(result=result)
            if loss is not None:
                if self.has_hook(ModelExecutorHookPoint.MODEL_BACKWARD):
                    self.exec_hooks(ModelExecutorHookPoint.MODEL_BACKWARD, loss=loss)
                else:
                    loss.backward()

            self.exec_hooks(
                ModelExecutorHookPoint.AFTER_BATCH,
                batch_index=batch_index,
                input_features=result["input_features"],
                epoch=epoch,
                result=result,
                **batch,
            )
            if in_training:
                if self.has_hook(ModelExecutorHookPoint.OPTIMIZER_STEP):
                    self.set_data("step_skipped", False)
                    self.exec_hooks(ModelExecutorHookPoint.OPTIMIZER_STEP)
                    step_skipped: bool = self.get_data("step_skipped")
                else:
                    optimizer.step()
                    step_skipped: bool = False
                lr_scheduler = self.get_lr_scheduler()
                if HyperParameter.lr_scheduler_step_after_batch(lr_scheduler):
                    get_logger().debug("adjust lr after batch")
                    lr_scheduler.step()

                self.exec_hooks(
                    ModelExecutorHookPoint.AFTER_OPTIMIZER_STEP,
                    epoch=epoch,
                    batch_index=batch_index,
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
