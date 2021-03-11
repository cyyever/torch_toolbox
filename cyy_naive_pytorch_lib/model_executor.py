import copy
import os
from typing import Callable, Dict, List, Union

import torch

from dataset_collection import DatasetCollection
from device import get_device, put_data_to_device
from hyper_parameter import HyperParameter
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
        self.__model_with_loss = copy.deepcopy(model_with_loss)
        self.__model_with_loss.set_model_mode(phase)
        self.__dataset_collection: DatasetCollection = dataset_collection
        self.__phase = phase
        self.__hyper_parameter = hyper_parameter
        self.__device = get_device()
        self.__cuda_stream = None
        self.__data: dict = dict()
        self.__callbacks: Dict[
            ModelExecutorCallbackPoint, List[Union[Callable, Dict[str, Callable]]]
        ] = dict()

    @property
    def phase(self):
        return self.__phase

    @property
    def dataset(self):
        return self.dataset_collection.get_dataset(phase=self.__phase)

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
        return self.model_with_loss.model

    def get_data(self, key: str):
        assert key in self.__data
        return self.__data.get(key)

    def set_data(self, key: str, value):
        self.__data[key] = value

    def remove_data(self, key: str):
        self.__data.pop(key, None)

    def has_data(self, key: str):
        return key in self.__data

    def exec_callbacks(self, cb_point: ModelExecutorCallbackPoint, *args, **kwargs):
        for o in self.__callbacks.get(cb_point, []):
            cbs = [o]
            if isinstance(o, dict):
                cbs = o.values()
            for cb in cbs:
                cb(*args, **kwargs)

    def has_callback(
        self,
        cb_point: ModelExecutorCallbackPoint,
    ):
        return cb_point in self.__callbacks

    def prepend_callback(
        self,
        cb_point: ModelExecutorCallbackPoint,
        cb: Union[Callable, Dict[str, Callable]],
    ):
        if cb_point not in self.__callbacks:
            self.__callbacks[cb_point] = [cb]
        else:
            self.__callbacks[cb_point].insert(0, cb)

    def add_callback(
        self,
        cb_point: ModelExecutorCallbackPoint,
        cb: Union[Callable, Dict[str, Callable]],
    ):
        if cb_point not in self.__callbacks:
            self.__callbacks[cb_point] = [cb]
        else:
            self.__callbacks[cb_point].append(cb)

    def add_named_callback(
        self, cb_point: ModelExecutorCallbackPoint, name: str, cb: Callable
    ):
        self.add_callback(cb_point, {name: cb})

    def prepend_named_callback(
        self, cb_point: ModelExecutorCallbackPoint, name: str, cb: Callable
    ):
        self.prepend_callback(cb_point, {name: cb})

    def remove_callback(self, name: str, cb_point: ModelExecutorCallbackPoint = None):
        for cur_cb_point, callbacks in self.__callbacks.items():
            if cb_point is not None and cur_cb_point != cb_point:
                continue
            for idx, cb in enumerate(callbacks):
                if isinstance(cb, dict):
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
        if self.__cuda_stream is None:
            if self.device.type.lower() == "cuda":
                self.__cuda_stream = torch.cuda.Stream(device=self.device)
        return self.__cuda_stream

    def set_hyper_parameter(self, hyper_parameter):
        self.__hyper_parameter = hyper_parameter

    @property
    def hyper_parameter(self):
        return self.__hyper_parameter

    def set_model(self, model: torch.nn.Module):
        self.model_with_loss.set_model(model)

    def load_model(self, model_path):
        self.set_model(torch.load(model_path, map_location=self.device))

    def save_model(self, save_dir, model_name="model.pt"):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model, os.path.join(save_dir, model_name))

    def decode_batch(self, batch, device=None):
        if device is None:
            device = self.device
        sample_inputs = put_data_to_device(batch[0], device)
        sample_targets = put_data_to_device(batch[1], device)
        sample_indices = None
        if len(batch) >= 3:
            sample_indices = [idx.data.item() for idx in batch[2]]
        return (sample_inputs, sample_targets, sample_indices)

    @staticmethod
    def get_batch_size(batch):
        if isinstance(batch, torch.Tensor):
            return batch.shape[0]
        if isinstance(batch, list):
            return len(batch)
        raise RuntimeError("invalid tensors:" + str(batch))
