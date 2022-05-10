import copy
import os
from typing import Callable, Optional

import torch
from cyy_naive_lib.log import get_logger

from cyy_torch_toolbox.dataloader import get_dataloader
from cyy_torch_toolbox.dataset import get_dataset_size
from cyy_torch_toolbox.dataset_collection import DatasetCollection
from cyy_torch_toolbox.device import get_device
from cyy_torch_toolbox.hooks.model_executor_logger import ModelExecutorLogger
from cyy_torch_toolbox.hooks.profiler import Profiler
from cyy_torch_toolbox.hyper_parameter import HyperParameter
from cyy_torch_toolbox.metric_visualizers.metric_tensorboard import \
    MetricTensorBoard
from cyy_torch_toolbox.metric_visualizers.performance_metric_logger import \
    PerformanceMetricLogger
from cyy_torch_toolbox.metrics.performance_metric import PerformanceMetric
from cyy_torch_toolbox.ml_type import MachineLearningPhase
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
        self.__performance_metric = PerformanceMetric(self._model_with_loss.model_type)
        self.append_hook(self.__performance_metric)
        self.__performance_metric_logger = PerformanceMetricLogger()
        self.append_hook(self.__performance_metric_logger)
        self.__metric_tb: MetricTensorBoard = MetricTensorBoard()
        self.append_hook(self.__metric_tb)
        self.debugging_mode = False
        self.profiling_mode = False
        self.__profiler = None
        self.__save_dir: Optional[str] = None

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

    def set_save_dir(self, save_dir: str):
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
    def dataset_size(self):
        return get_dataset_size(self.dataset)

    def transform_dataset(self, transformer: Callable):
        self.dataset_collection.transform_dataset(self.phase, transformer)

    @property
    def dataloader(self):
        if self.__dataloader is None:
            self.__dataloader = get_dataloader(
                self.dataset_collection,
                self._model_with_loss.model_type,
                self.__phase,
                self.__hyper_parameter,
                device=self.device,
                stream=self.cuda_stream,
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

    def copy_model_with_loss(self, deepcopy=True):
        self._wait_stream()
        if deepcopy:
            return copy.deepcopy(self._model_with_loss)
        return copy.copy(self._model_with_loss)

    def _prepare_execution(self, **kwargs):
        self.clear_data()
        for name in dir(self):
            if isinstance(getattr(type(self), name, None), property):
                continue
            attr = getattr(self, name)
            if hasattr(attr, "_is_cyy_torch_toolbox_metric"):
                attr.clear_metric()

        if self.profiling_mode:
            get_logger().warning("train in profiling mode")
            if self.__profiler is None:
                self.__profiler = Profiler()
                self.append_hook(self.__profiler)
            else:
                self.enable_hook(self.__profiler)
        else:
            if self.__profiler is not None:
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

    def set_model_with_loss(self, model_with_loss: ModelWithLoss):
        self._wait_stream()
        self._model_with_loss = model_with_loss

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
        # if isinstance(batch, list):
        #     return len(batch)

    def offload_from_gpu(self):
        self._wait_stream()
        self._model_with_loss.offload_from_gpu()
        if self.__dataloader is not None:
            del self.__dataloader
            self.__dataloader = None
        torch.cuda.empty_cache()

    @classmethod
    def decode_batch(cls, batch):
        batch_size = None
        if isinstance(batch, dict):
            batch_size = batch["size"]
            batch = batch["content"]
        sample_inputs = batch[0]
        sample_targets = batch[1]
        if len(batch) >= 3:
            return (batch_size, sample_inputs, sample_targets, batch[2])
        return (batch_size, sample_inputs, sample_targets, {})
