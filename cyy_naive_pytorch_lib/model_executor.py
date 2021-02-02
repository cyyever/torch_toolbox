import copy
# import logging
# import os
from enum import IntEnum, auto
from typing import Callable, Dict, List, Union

import torch

from dataset_collection import DatasetCollection
from device import get_device, put_data_to_device
from hyper_parameter import HyperParameter
from model_loss import ModelWithLoss


class ModelExecutorCallbackPoint(IntEnum):
    BEFORE_TRAINING = auto()
    AFTER_TRAINING = auto()
    BEFORE_EPOCH = auto()
    AFTER_EPOCH = auto()
    OPTIMIZER_STEP = auto()
    BEFORE_BATCH = auto()
    AFTER_BATCH = auto()


class ModelExecutor:
    def __init__(
        self,
        model_with_loss: ModelWithLoss,
        dataset_collection: DatasetCollection,
        hyper_parameter: HyperParameter,
    ):
        self.__model_with_loss = copy.deepcopy(model_with_loss)
        self.__dataset_collection: DatasetCollection = dataset_collection
        self.__hyper_parameter = hyper_parameter
        self.__device = get_device()
        self.__data: dict = dict()
        self.__callbacks: Dict[
            ModelExecutorCallbackPoint, List[Union[Callable, Dict[str, Callable]]]
        ] = dict()

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

    def remove_callback(self, cb_point: ModelExecutorCallbackPoint, name: str):
        if cb_point not in self.__callbacks:
            return
        for idx, cb in enumerate(self.__callbacks[cb_point]):
            if isinstance(cb, dict):
                cb.pop(name, None)
                self.__callbacks[cb_point][idx] = cb

    @property
    def dataset_collection(self):
        return self.__dataset_collection

    @property
    def device(self):
        return self.__device

    def set_device(self, device):
        self.__device = device

    def set_hyper_parameter(self, hyper_parameter):
        self.__hyper_parameter = hyper_parameter

    @property
    def hyper_parameter(self):
        return self.__hyper_parameter

    def set_model(self, model: torch.nn.Module):
        self.model_with_loss.set_model(model)

    def load_model(self, model_path):
        self.set_model(torch.load(model_path, map_location=self.device))

    def decode_batch(self, batch):
        instance_inputs = put_data_to_device(batch[0], self.device)
        instance_targets = put_data_to_device(batch[1], self.device)
        instance_indices = None
        if len(batch) >= 3:
            instance_indices = [idx.data.item() for idx in batch[2]]
        return (instance_inputs, instance_targets, instance_indices)
