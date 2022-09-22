import copy
import functools
from enum import IntEnum, auto
from typing import Callable, Optional, Union

import torch
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.reflection import call_fun, get_class_attrs

from cyy_torch_toolbox.algorithm.lr_finder import LRFinder
from cyy_torch_toolbox.data_structure.torch_thread_task_queue import \
    TorchThreadTaskQueue


def determin_learning_rate(task, *args):
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


class HyperParameter:
    def __init__(
        self,
        epoch: int,
        batch_size: int,
        learning_rate: Union[float, HyperParameterAction],
        weight_decay: float,
    ):
        self.__epoch = epoch
        self.__batch_size = batch_size
        self.__learning_rate = learning_rate
        self.__weight_decay = weight_decay
        self.__momentum = 0.9
        self.__lr_scheduler_factory: Optional[Callable] = None
        self.__optimizer_factory: Optional[Callable] = None

    # def __getstate__(self):
    #     # capture what is normally pickled
    #     state = self.__dict__.copy()
    #     state["_HyperParameter__lr_scheduler_factory"] = None
    #     return state

    @property
    def epoch(self):
        return self.__epoch

    def set_epoch(self, epoch):
        self.__epoch = epoch

    @property
    def batch_size(self):
        return self.__batch_size

    def set_batch_size(self, batch_size):
        self.__batch_size = batch_size

    def get_learning_rate(self, trainer):
        if isinstance(self.__learning_rate, HyperParameterAction):
            task_queue = TorchThreadTaskQueue(worker_fun=determin_learning_rate)
            device = trainer.device
            trainer.offload_from_gpu()
            task_queue.add_task((copy.deepcopy(trainer), device))
            self.__learning_rate = task_queue.get_result()
            trainer.set_device(device)
            task_queue.stop()
        return self.__learning_rate

    def set_learning_rate(
        self, learning_rate: Union[float, HyperParameterAction]
    ) -> None:
        self.__learning_rate = learning_rate

    @property
    def weight_decay(self):
        return self.__weight_decay

    def set_weight_decay(self, weight_decay):
        self.__weight_decay = weight_decay

    @property
    def momentum(self):
        return self.__momentum

    def set_momentum(self, momentum):
        self.__momentum = momentum

    def set_lr_scheduler_factory(self, lr_scheduler_factory: Callable) -> None:
        self.__lr_scheduler_factory = lr_scheduler_factory

    def __get_iterations_per_epoch(self, training_dataset_size):
        if self.batch_size == 1:
            return training_dataset_size
        return (training_dataset_size + self.batch_size - 1) // self.batch_size

    def get_lr_scheduler(self, trainer):
        assert self.__lr_scheduler_factory is not None
        return self.__lr_scheduler_factory(self, trainer)

    @staticmethod
    def lr_scheduler_step_after_batch(lr_scheduler):
        return isinstance(lr_scheduler, torch.optim.lr_scheduler.OneCycleLR)

    @classmethod
    def get_lr_scheduler_factory(cls, name, dataset_name=None, **kwargs):
        return functools.partial(
            cls.default_lr_scheduler_factory,
            name=name,
            dataset_name=dataset_name,
            kwargs=kwargs,
        )

    @classmethod
    def default_lr_scheduler_factory(
        cls, hyper_parameter, trainer, name, dataset_name, kwargs
    ):
        optimizer = trainer.get_optimizer()
        training_dataset_size = trainer.dataset_size
        full_kwargs: dict = {}
        full_kwargs["optimizer"] = optimizer
        if name == "ReduceLROnPlateau":
            patience = min(10, hyper_parameter.epoch + 9 // 10)
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
            full_kwargs["max_lr"] = 10 * hyper_parameter.get_learning_rate(trainer)
            full_kwargs[
                "total_steps"
            ] = hyper_parameter.epoch * hyper_parameter.__get_iterations_per_epoch(
                training_dataset_size
            )
            full_kwargs["anneal_strategy"] = "linear"
            full_kwargs["three_phase"] = True
            full_kwargs.update(kwargs)
            return torch.optim.lr_scheduler.OneCycleLR(**full_kwargs)
        if name == "CosineAnnealingLR":
            full_kwargs["T_max"] = hyper_parameter.epoch
            full_kwargs.update(kwargs)
            return torch.optim.lr_scheduler.CosineAnnealingLR(**full_kwargs)
        if name == "MultiStepLR":
            full_kwargs["T_max"] = hyper_parameter.epoch
            full_kwargs["milestones"] = [30, 80]
            full_kwargs.update(kwargs)
            return torch.optim.lr_scheduler.MultiStepLR(**full_kwargs)
        fun = getattr(torch.optim.lr_scheduler, name)
        if fun is not None:
            full_kwargs.update(kwargs)
            return fun(**full_kwargs)

        raise RuntimeError("unknown learning rate scheduler:" + name)

    def set_optimizer_factory(self, optimizer_factory: Callable) -> None:
        self.__optimizer_factory = optimizer_factory

    @staticmethod
    def get_optimizer_names():
        return sorted(HyperParameter.__get_optimizer_classes().keys())

    @staticmethod
    def get_lr_scheduler_names() -> list:
        return ["ReduceLROnPlateau", "OneCycleLR", "CosineAnnealingLR", "MultiStepLR"]

    @classmethod
    def get_optimizer_factory(cls, name: str):
        optimizer_class = cls.__get_optimizer_classes().get(name, None)
        if optimizer_class is None:
            raise RuntimeError(
                f"unknown optimizer:{name}, supported names are:"
                + str(HyperParameter.get_optimizer_names())
            )
        return optimizer_class

    def get_optimizer(self, trainer):
        assert self.__optimizer_factory is not None
        kwargs: dict = {
            "params": trainer.model.parameters(),
            "lr": self.get_learning_rate(trainer),
            "momentum": self.momentum,
            "weight_decay": self.weight_decay / trainer.dataset_size,
            "foreach": True,
        }
        return call_fun(self.__optimizer_factory, kwargs)

    @staticmethod
    def __get_optimizer_classes():
        return get_class_attrs(
            torch.optim,
            filter_fun=lambda _, v: issubclass(v, torch.optim.Optimizer),
        )

    def __str__(self):
        s = (
            "epoch:"
            + str(self.epoch)
            + " batch_size:"
            + str(self.batch_size)
            + " learning_rate:"
            + str(self.__learning_rate)
            + " weight_decay:"
            + str(self.weight_decay)
        )
        return s


def get_recommended_hyper_parameter(
    dataset_name: str, model_name: str
) -> Optional[HyperParameter]:
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
            epoch=350, batch_size=64, learning_rate=0.1, weight_decay=1
        )
    hyper_parameter.set_lr_scheduler_factory(
        HyperParameter.get_lr_scheduler_factory("ReduceLROnPlateau", dataset_name)
    )
    hyper_parameter.set_optimizer_factory(HyperParameter.get_optimizer_factory("SGD"))
    return hyper_parameter


class HyperParameterConfig:
    def __init__(self):
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
            hyper_parameter.set_epoch(self.epoch)
        if self.batch_size is not None:
            hyper_parameter.set_batch_size(self.batch_size)
        if self.learning_rate is not None:
            self.find_learning_rate = False
        if self.learning_rate is not None:
            hyper_parameter.set_learning_rate(self.learning_rate)
        if self.find_learning_rate:
            hyper_parameter.set_learning_rate(HyperParameterAction.FIND_LR)
        if self.momentum is not None:
            hyper_parameter.set_momentum(self.momentum)
        if self.weight_decay is not None:
            hyper_parameter.set_weight_decay(self.weight_decay)
        if self.optimizer_name is not None:
            hyper_parameter.set_optimizer_factory(
                HyperParameter.get_optimizer_factory(self.optimizer_name)
            )
        if self.learning_rate_scheduler is not None:
            hyper_parameter.set_lr_scheduler_factory(
                HyperParameter.get_lr_scheduler_factory(
                    self.learning_rate_scheduler,
                    **self.learning_rate_scheduler_kwargs,
                )
            )
        return hyper_parameter
