import copy
import functools
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Any, Callable

import torch
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.reflection import call_fun, get_class_attrs

from ..data_structure.torch_thread_task_queue import TorchThreadTaskQueue
from .lr_finder import LRFinder


def determine_learning_rate(task: Any, **kwargs: Any) -> float:
    tmp_trainer, device = task
    tmp_trainer.set_device(device)
    tmp_trainer.disable_stripable_hooks()
    lr_finder = LRFinder()
    get_logger().warning("register lr_finder")
    tmp_trainer.prepend_hook(lr_finder)
    tmp_trainer.train()
    get_logger().warning(
        "suggested_learning_rate is %s", lr_finder.suggested_learning_rate
    )
    return lr_finder.suggested_learning_rate


class HyperParameterAction(IntEnum):
    FIND_LR = auto()


@dataclass(kw_only=True)
class HyperParameter:
    epoch: int
    learning_rate: float | HyperParameterAction = HyperParameterAction.FIND_LR
    batch_size: int = 8
    weight_decay: float = 0
    momentum: float = 0.9
    _lr_scheduler_factory: None | Callable = None
    _optimizer_factory: None | Callable = None

    def get_learning_rate(self, trainer: Any) -> float:
        if isinstance(self.learning_rate, HyperParameterAction):
            task_queue = TorchThreadTaskQueue(worker_fun=determine_learning_rate)
            device = trainer.device
            trainer.offload_from_device()
            task_queue.add_task((copy.deepcopy(trainer), device))
            self.learning_rate = task_queue.get_data()[0]
            trainer.set_device(device)
            task_queue.stop()
        return self.learning_rate

    def set_lr_scheduler_factory(
        self, name: str, dataset_name: None | str = None, **kwargs: Any
    ) -> None:
        self._lr_scheduler_factory = functools.partial(
            self.__get_lr_scheduler_factory,
            name=name,
            dataset_name=dataset_name,
            kwargs=kwargs,
        )

    def get_iterations_per_epoch(self, training_dataset_size):
        if self.batch_size == 1:
            return training_dataset_size
        return (training_dataset_size + self.batch_size - 1) // self.batch_size

    def get_lr_scheduler(self, trainer):
        assert self._lr_scheduler_factory is not None
        return self._lr_scheduler_factory(trainer)

    @staticmethod
    def lr_scheduler_step_after_batch(lr_scheduler):
        return isinstance(lr_scheduler, torch.optim.lr_scheduler.OneCycleLR)

    def __get_lr_scheduler_factory(
        self, trainer: Any, name: str, dataset_name: str, kwargs: dict
    ) -> Any:
        optimizer = trainer.get_optimizer()
        training_dataset_size = trainer.dataset_size
        full_kwargs: dict = {}
        full_kwargs["optimizer"] = optimizer
        if name == "ReduceLROnPlateau":
            patience = min(10, self.epoch + 9 // 10)
            if dataset_name == "CIFAR10":
                patience = 2
            full_kwargs["patience"] = patience
            full_kwargs["factor"] = 0.1
            full_kwargs["verbose"] = True
            full_kwargs.update(kwargs)
            get_logger().debug(
                "ReduceLROnPlateau patience is %s", full_kwargs["patience"]
            )
            return torch.optim.lr_scheduler.ReduceLROnPlateau(**full_kwargs)
        if name == "OneCycleLR":
            full_kwargs["pct_start"] = 0.4
            full_kwargs["max_lr"] = 10 * self.get_learning_rate(trainer)
            full_kwargs["total_steps"] = self.epoch * self.get_iterations_per_epoch(
                training_dataset_size
            )
            full_kwargs["anneal_strategy"] = "linear"
            full_kwargs["three_phase"] = True
            full_kwargs.update(kwargs)
            return torch.optim.lr_scheduler.OneCycleLR(**full_kwargs)
        if name == "CosineAnnealingLR":
            full_kwargs["T_max"] = self.epoch
            full_kwargs.update(kwargs)
            return torch.optim.lr_scheduler.CosineAnnealingLR(**full_kwargs)
        if name == "MultiStepLR":
            full_kwargs["T_max"] = self.epoch
            full_kwargs["milestones"] = [30, 80]
            full_kwargs.update(kwargs)
            return torch.optim.lr_scheduler.MultiStepLR(**full_kwargs)
        fun = getattr(torch.optim.lr_scheduler, name)
        if fun is not None:
            full_kwargs.update(kwargs)
            return fun(**full_kwargs)

        raise RuntimeError("unknown learning rate scheduler:" + name)

    def set_optimizer_factory(self, name: str) -> None:
        optimizer_class = self.__get_optimizer_classes().get(name, None)
        if optimizer_class is None:
            raise RuntimeError(
                f"unknown optimizer:{name}, supported names are:"
                + str(HyperParameter.get_optimizer_names())
            )
        self._optimizer_factory = optimizer_class

    @staticmethod
    def get_optimizer_names():
        return sorted(HyperParameter.__get_optimizer_classes().keys())

    @staticmethod
    def get_lr_scheduler_names() -> list:
        return ["ReduceLROnPlateau", "OneCycleLR", "CosineAnnealingLR", "MultiStepLR"]

    def get_optimizer(self, trainer: Any) -> Any:
        assert self._optimizer_factory is not None
        foreach = not torch.backends.mps.is_available()
        kwargs: dict = {
            "params": trainer.model.parameters(),
            "lr": self.get_learning_rate(trainer),
            "momentum": self.momentum,
            "weight_decay": self.weight_decay / trainer.dataset_size,
            "foreach": foreach,
        }
        return call_fun(self._optimizer_factory, kwargs)

    @staticmethod
    def __get_optimizer_classes():
        return get_class_attrs(
            torch.optim,
            filter_fun=lambda _, v: issubclass(v, torch.optim.Optimizer),
        )

    def __str__(self) -> str:
        s = (
            "epoch:"
            + str(self.epoch)
            + " batch_size:"
            + str(self.batch_size)
            + " learning_rate:"
            + str(self.learning_rate)
            + " weight_decay:"
            + str(self.weight_decay)
        )
        return s


def get_recommended_hyper_parameter(
    dataset_name: str, model_name: str
) -> None | HyperParameter:
    """
    Given dataset and model, return a set of recommended hyper parameters
    """

    hyper_parameter = None

    if dataset_name == "MNIST":
        hyper_parameter = HyperParameter(
            epoch=50, batch_size=64, learning_rate=0.01, weight_decay=1
        )
    elif dataset_name == "FashionMNIST" and model_name.lower() == "LeNet5".lower():
        hyper_parameter = HyperParameter(
            epoch=50, batch_size=64, learning_rate=0.01, weight_decay=1
        )
    elif dataset_name == "CIFAR10":
        hyper_parameter = HyperParameter(
            epoch=350, batch_size=64, learning_rate=0.1, weight_decay=1
        )
    elif dataset_name == "CIFAR100":
        hyper_parameter = HyperParameter(
            epoch=350,
            batch_size=64,
            learning_rate=HyperParameterAction.FIND_LR,
            weight_decay=1,
        )
    elif dataset_name == "SVHN":
        hyper_parameter = HyperParameter(
            epoch=50, batch_size=4, learning_rate=0.0001, weight_decay=1
        )
    else:
        hyper_parameter = HyperParameter(
            epoch=350,
            batch_size=64,
            learning_rate=HyperParameterAction.FIND_LR,
            weight_decay=0,
        )
    hyper_parameter.set_lr_scheduler_factory(
        name="ReduceLROnPlateau", dataset_name=dataset_name
    )
    hyper_parameter.set_optimizer_factory("Adam")
    return hyper_parameter


class HyperParameterConfig:
    def __init__(self) -> None:
        self.epoch = None
        self.batch_size = None
        self.find_learning_rate = True
        self.learning_rate = None
        self.learning_rate_scheduler = None
        self.learning_rate_scheduler_kwargs = {}
        self.momentum = None
        self.weight_decay = None
        self.optimizer_name = None

    def create_hyper_parameter(self, dataset_name, model_name):
        hyper_parameter = get_recommended_hyper_parameter(dataset_name, model_name)

        if self.epoch is not None:
            hyper_parameter.epoch = self.epoch
        if self.batch_size is not None:
            hyper_parameter.batch_size = self.batch_size
        if self.learning_rate is not None:
            self.find_learning_rate = False
            hyper_parameter.learning_rate = self.learning_rate
        elif self.find_learning_rate:
            hyper_parameter.learning_rate = HyperParameterAction.FIND_LR
        if self.momentum is not None:
            hyper_parameter.momentum = self.momentum
        if self.weight_decay is not None:
            hyper_parameter.weight_decay = self.weight_decay
        if self.optimizer_name is not None:
            hyper_parameter.set_optimizer_factory(self.optimizer_name)
        if self.learning_rate_scheduler is not None:
            hyper_parameter.set_lr_scheduler_factory(
                name=self.learning_rate_scheduler,
                **self.learning_rate_scheduler_kwargs,
            )
        return hyper_parameter
