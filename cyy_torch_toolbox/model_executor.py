import copy
import os
from typing import Callable, Dict, List, Optional

import torch

from dataset_collection import DatasetCollection
from device import get_device
from hook import Hook
from hooks.model_executor_logger import ModelExecutorLogger
from hyper_parameter import HyperParameter
from metric_visualizers.metric_tensorboard import MetricTensorBoard
from metric_visualizers.performance_metric_logger import \
    PerformanceMetricLogger
from metrics.performance_metric import PerformanceMetric
from ml_type import MachineLearningPhase, ModelExecutorHookPoint
from model_with_loss import ModelWithLoss


class ModelExecutor:
    def __init__(
        self,
        model_with_loss: ModelWithLoss,
        dataset_collection: DatasetCollection,
        phase: MachineLearningPhase,
        hyper_parameter: HyperParameter,
        save_dir=None,
    ):
        self._model_with_loss = model_with_loss
        self.__dataset_collection: DatasetCollection = dataset_collection
        self.__phase = phase
        self.__hyper_parameter = hyper_parameter
        self.__device = get_device()
        self.__cuda_stream = None
        self.__data: dict = dict()
        self.__hooks: Dict[ModelExecutorHookPoint, List[Dict[str, Callable]]] = dict()
        self.__stripable_hooks: set = set()
        self.__disabled_hooks: set = set()

        self.__metric_tb: MetricTensorBoard = MetricTensorBoard()
        self.__logger = ModelExecutorLogger()
        self.append_hook(self.__logger)
        self.__performance_metric = PerformanceMetric(self._model_with_loss.model_type)
        self.append_hook(self.__performance_metric)
        self.__performance_metric_logger = PerformanceMetricLogger()
        self.append_hook(self.__performance_metric_logger)
        self.debugging_mode = False
        self.__save_dir: Optional[str] = None
        if save_dir is not None:
            self.set_save_dir(save_dir)

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

    def transform_dataset(self, transformer: Callable):
        self.dataset_collection.transform_dataset(self.phase, transformer)

    @property
    def dataloader(self):
        return self.dataset_collection.get_dataloader(
            self.__phase, self.__hyper_parameter, device=self.device
        )

    @property
    def loss_fun(self):
        return self._model_with_loss.loss_fun

    @property
    def model(self) -> torch.nn.Module:
        self._wait_stream()
        return self._model_with_loss.model

    def copy_model_with_loss(self, deepcopy=True):
        self._wait_stream()
        if deepcopy:
            return copy.deepcopy(self._model_with_loss)
        return copy.copy(self._model_with_loss)

    def get_data(self, key: str, default_value=None):
        return self.__data.get(key, default_value)

    def set_data(self, key: str, value):
        self.__data[key] = value

    def remove_data(self, key: str):
        self.__data.pop(key, None)

    def has_data(self, key: str):
        return key in self.__data

    def _prepare_execution(self):
        self.__data.clear()
        for name in dir(self):
            attr = getattr(self, name)
            if hasattr(attr, "_is_cyy_torch_toolbox_metric"):
                attr.clear_metric()

    def exec_hooks(self, hook_point: ModelExecutorHookPoint, **kwargs):
        for hook in self.__hooks.get(hook_point, []):
            for name, fun in hook.items():
                if name not in self.__disabled_hooks:
                    fun(**kwargs)

    def has_hook(
        self,
        hook_point: ModelExecutorHookPoint,
    ):
        return hook_point in self.__hooks

    def hooks(self):
        return self.__hooks

    def disable_stripable_hooks(self):
        self.__disabled_hooks.update(self.__stripable_hooks)

    def enable_all_hooks(self):
        self.__disabled_hooks.clear()

    def append_named_hook(
        self,
        hook_point: ModelExecutorHookPoint,
        name: str,
        fun: Callable,
        stripable=False,
    ):
        self.insert_callback(-1, hook_point, name, fun, stripable)

    def insert_callback(
        self,
        pos,
        hook_point: ModelExecutorHookPoint,
        name: str,
        fun: Callable,
        stripable=False,
    ):
        if stripable:
            self.__stripable_hooks.add(name)
        data = {name: fun}
        if hook_point not in self.__hooks:
            self.__hooks[hook_point] = [data]
        else:
            for d in self.__hooks[hook_point]:
                if name in d:
                    raise RuntimeError(name + " has registered")
            if pos < 0:
                self.__hooks[hook_point].append(data)
            else:
                self.__hooks[hook_point].insert(pos, data)

    def insert_hook(self, pos, hook: Hook):
        for hook_point, name, fun in hook.yield_hooks():
            self.insert_callback(pos, hook_point, name, fun, hook.stripable)

    def append_hook(self, hook: Hook):
        self.insert_hook(-1, hook)

    def prepend_hook(self, hook: Hook):
        self.insert_hook(0, hook)

    def enable_hook(self, hook: Hook):
        for name in hook.yield_hook_names():
            if name in self.__disabled_hooks:
                self.__disabled_hooks.remove(name)

    def disable_hook(self, hook: Hook):
        for name in hook.yield_hook_names():
            self.__disabled_hooks.add(name)

    def remove_hook(self, name: str, hook_point: ModelExecutorHookPoint = None):
        for cur_hook_point, hooks in self.__hooks.items():
            if hook_point is not None and cur_hook_point != hook_point:
                continue
            for idx, hook in enumerate(hooks):
                hook.pop(name, None)
                self.__hooks[cur_hook_point][idx] = hook

    @property
    def dataset_collection(self) -> DatasetCollection:
        return self.__dataset_collection

    @property
    def device(self):
        return self.__device

    def set_device(self, device):
        self._wait_stream()
        self.__device = device
        self.__cuda_stream = None

    def set_stream(self, stream):
        self._wait_stream()
        self.__cuda_stream = stream

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

    @property
    def hyper_parameter(self):
        return self.__hyper_parameter

    def set_model(self, model: torch.nn.Module):
        self._model_with_loss.set_model(model)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def get_batch_size(self, batch):
        if isinstance(batch, tuple):
            return self.get_batch_size(batch[0])
        if isinstance(batch, torch.Tensor):
            return batch.shape[0]
        if isinstance(batch, list):
            return len(batch)
        raise RuntimeError("invalid batch:" + str(batch))

    def offload_from_gpu(self):
        self.model.cpu()
        torch.cuda.empty_cache()
