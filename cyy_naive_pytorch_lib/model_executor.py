import copy
from typing import Callable, Dict, List

import torch
from cyy_naive_lib.log import get_logger

from dataset_collection import DatasetCollection
from device import get_device
from hooks.model_executor_logger import ModelExecutorLogger
from hyper_parameter import HyperParameter
from metric_visualizers.performance_metric_logger import \
    PerformanceMetricLogger
from metrics.performance_metric import PerformanceMetric
from ml_type import MachineLearningPhase, ModelExecutorCallbackPoint
from model_with_loss import ModelWithLoss


class ModelExecutor:
    def __init__(
        self,
        model_with_loss: ModelWithLoss,
        dataset_collection: DatasetCollection,
        phase: MachineLearningPhase,
        hyper_parameter: HyperParameter,
    ):
        self.__model_with_loss = model_with_loss
        self.__dataset_collection: DatasetCollection = dataset_collection
        self.__phase = phase
        self.__hyper_parameter = hyper_parameter
        self.__device = get_device()
        self.__cuda_stream = None
        self.__data: dict = dict()
        self.__callbacks: Dict[
            ModelExecutorCallbackPoint, List[Dict[str, Callable]]
        ] = dict()
        self.__stripable_callbacks: set = set()
        self.__disabled_callbacks: set = set()

        self.__logger = ModelExecutorLogger()
        self.__logger.append_to_model_executor(self)
        self.__performance_metric = PerformanceMetric()
        self.__performance_metric.append_to_model_executor(self)
        self.__performance_metric_logger = PerformanceMetricLogger()
        self.__performance_metric_logger.append_to_model_executor(self)
        self.debugging_mode = False

    @property
    def phase(self):
        return self.__phase

    @property
    def performance_metric(self):
        return self.__performance_metric

    @property
    def performance_metric_logger(self):
        return self.__performance_metric_logger

    def remove_logger(self):
        if self.__logger is not None:
            self.__logger.remove_from_model_executor(self)
        self.__logger = None

    def remove_performance_metric_logger(self):
        if self.__performance_metric_logger is not None:
            self.__performance_metric_logger.remove_from_model_executor(self)
        self.__performance_metric_logger = None

    @property
    def dataset(self):
        return self.dataset_collection.get_dataset(phase=self.__phase)

    def transform_dataset(self, transformer: Callable):
        self.dataset_collection.transform_dataset(self.phase, transformer)

    @property
    def dataloader(self):
        return self.dataset_collection.get_dataloader(
            self.__phase,
            self.__hyper_parameter,
        )

    @property
    def model_with_loss(self):
        return self.__model_with_loss

    @property
    def model(self) -> torch.nn.Module:
        self.__wait_stream()
        return self.model_with_loss.model

    def copy_model_with_loss(self, deepcopy=True):
        self.__wait_stream()
        if deepcopy:
            return copy.deepcopy(self.model_with_loss)
        return copy.copy(self.model_with_loss)

    def get_data(self, key: str, default_value=None):
        assert key in self.__data
        return self.__data.get(key, default_value)

    def set_data(self, key: str, value):
        self.__data[key] = value

    def remove_data(self, key: str):
        self.__data.pop(key, None)

    def has_data(self, key: str):
        return key in self.__data

    def exec_callbacks(self, cb_point: ModelExecutorCallbackPoint, *args, **kwargs):
        for o in self.__callbacks.get(cb_point, []):
            for name, cb in o.items():
                if name in self.__disabled_callbacks:
                    continue
                get_logger().debug("call %s", name)
                cb(*args, **kwargs)

    def has_callback(
        self,
        cb_point: ModelExecutorCallbackPoint,
    ):
        return cb_point in self.__callbacks

    def callbacks(self):
        return self.__callbacks

    def disable_stripable_callbacks(self):
        self.__disabled_callbacks.update(self.__stripable_callbacks)

    def enable_all_callbacks(self):
        self.__disabled_callbacks.clear()

    def append_callback(
        self, cb_point: ModelExecutorCallbackPoint, name: str, cb: Callable
    ):
        data = {name: cb}
        if cb_point not in self.__callbacks:
            self.__callbacks[cb_point] = [data]
        else:
            for d in self.__callbacks[cb_point]:
                if name in d:
                    raise RuntimeError(name + " has registered")
            self.__callbacks[cb_point].append(data)

    def set_stripable_callback(self, name: str):
        self.__stripable_callbacks.add(name)

    def prepend_before_other_callback(
        self,
        cb_point: ModelExecutorCallbackPoint,
        name: str,
        cb: Callable,
        other_name: str,
    ):
        data = {name: cb}
        assert cb_point in self.__callbacks
        for idx, other_data in enumerate(self.__callbacks[cb_point]):
            if other_name in other_data:
                self.__callbacks[cb_point].insert(idx, data)
                return
        raise RuntimeError("unknown callback:" + other_name)

    def prepend_callback(
        self,
        cb_point: ModelExecutorCallbackPoint,
        name: str,
        cb: Callable,
    ):
        data = {name: cb}
        if cb_point not in self.__callbacks:
            self.__callbacks[cb_point] = [data]
        else:
            for d in self.__callbacks[cb_point]:
                if name in d:
                    raise RuntimeError(name + " has registered")
            self.__callbacks[cb_point].insert(0, data)

    def remove_callback(self, name: str, cb_point: ModelExecutorCallbackPoint = None):
        for cur_cb_point, callbacks in self.__callbacks.items():
            if cb_point is not None and cur_cb_point != cb_point:
                continue
            for idx, cb in enumerate(callbacks):
                cb.pop(name, None)
                self.__callbacks[cur_cb_point][idx] = cb

    @property
    def dataset_collection(self) -> DatasetCollection:
        return self.__dataset_collection

    @property
    def device(self):
        return self.__device

    def set_device(self, device):
        self.__device = device
        self.__cuda_stream = None

    @property
    def cuda_stream(self):
        if self.__cuda_stream is None and "cuda" in self.device.type.lower():
            self.__cuda_stream = torch.cuda.Stream(device=self.device)
        return self.__cuda_stream

    def __wait_stream(self):
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
        self.model_with_loss.set_model(model)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def decode_batch(self, batch, device=None):
        if device is None:
            device = self.device
        sample_inputs = batch[0]
        sample_targets = batch[1]
        if len(batch) == 3:
            return (sample_inputs, sample_targets, batch[2])
        return (sample_inputs, sample_targets, {})

    def get_batch_size(self, targets):
        if isinstance(targets, torch.Tensor):
            return targets.shape[0]
        if isinstance(targets, list):
            return len(targets)
        raise RuntimeError("invalid targets:" + str(targets))
