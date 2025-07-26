import copy
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any

import torch
from cyy_naive_lib.log import log_debug
from cyy_naive_lib.reflection import call_fun, get_class_attrs

from .lr import lr_scheduler_step_after_batch
from .lr_finder import get_learning_rate
from .optimizer_repository import get_optimizer_names, global_optimizer_factory


class HyperParameterAction(StrEnum):
    FIND_LR = auto()


@dataclass(kw_only=True)
class HyperParameter:
    epoch: int = 350
    batch_size: int = 64
    learning_rate: HyperParameterAction | float = HyperParameterAction.FIND_LR
    learning_rate_scheduler_name: str = "ReduceLROnPlateau"
    learning_rate_scheduler_kwargs: dict = field(default_factory=lambda: {})
    optimizer_name: str = "AdamW"
    optimizer_kwargs: dict = field(default_factory=lambda: {"fake_weight_decay": 1.0})

    def get_iterations_per_epoch(self, dataset_size: int) -> int:
        if self.batch_size == 1:
            return dataset_size
        return (dataset_size + self.batch_size - 1) // self.batch_size

    def __get_learning_rate(self, trainer: Any | None = None) -> float:
        if isinstance(self.learning_rate, HyperParameterAction):
            self.learning_rate = get_learning_rate(trainer=trainer)
        return self.learning_rate

    def get_lr_scheduler(self, trainer) -> torch.optim.lr_scheduler.LRScheduler:
        name = self.learning_rate_scheduler_name
        optimizer = trainer.get_optimizer()
        default_kwargs: dict = {}
        default_kwargs["optimizer"] = optimizer
        fun = getattr(torch.optim.lr_scheduler, name)
        if fun is None:
            raise RuntimeError("Unknown learning rate scheduler:" + name)
        match self.learning_rate_scheduler_name:
            case "ReduceLROnPlateau":
                patience = min(10, self.epoch + 9 // 10)
                default_kwargs["patience"] = patience
                default_kwargs["factor"] = 0.1
                default_kwargs.update(self.learning_rate_scheduler_kwargs)
                log_debug(
                    "ReduceLROnPlateau patience is %s", default_kwargs["patience"]
                )
            case "OneCycleLR":
                default_kwargs["pct_start"] = 0.4
                default_kwargs["max_lr"] = 10 * self.__get_learning_rate(trainer)
                default_kwargs["total_steps"] = (
                    self.epoch * self.get_iterations_per_epoch(trainer.dataset_size)
                )
                default_kwargs["anneal_strategy"] = "linear"
                default_kwargs["three_phase"] = True
            case "CosineAnnealingLR":
                default_kwargs["T_max"] = self.epoch
            case "MultiStepLR":
                default_kwargs["milestones"] = [30, 80]
        fun = getattr(torch.optim.lr_scheduler, name)
        default_kwargs.update(self.learning_rate_scheduler_kwargs)
        return fun(**default_kwargs)

    @staticmethod
    def get_lr_scheduler_names() -> list[str]:
        return sorted(HyperParameter.__get_learning_rate_scheduler_classes().keys())

    def get_optimizer(
        self, trainer: Any, parameters: None | Iterable[torch.Tensor] = None
    ) -> Any:
        optimizer_class = global_optimizer_factory.get(self.optimizer_name)
        if optimizer_class is None:
            raise RuntimeError(
                f"unknown optimizer:{self.optimizer_name}, supported names are: {get_optimizer_names()}"
            )

        kwargs = copy.copy(self.optimizer_kwargs)
        if parameters is None:
            parameters = list(trainer.model.parameters())
        else:
            log_debug("pass provided parameters to optimizer")
        kwargs |= {
            "params": parameters,
            "lr": self.__get_learning_rate(trainer=trainer),
        }
        if "foreach" not in kwargs:
            kwargs["foreach"] = True
        kwargs.pop("learning_rate", None)
        if "fake_weight_decay" in kwargs and "weight_decay" not in kwargs:
            kwargs["weight_decay"] = (
                kwargs.pop("fake_weight_decay") / trainer.dataset_size
            )
        assert "weight_decay" in kwargs
        return call_fun(optimizer_class, kwargs)

    @staticmethod
    def __get_learning_rate_scheduler_classes() -> dict:
        return get_class_attrs(
            torch.optim.lr_scheduler,
            filter_fun=lambda _, v: issubclass(v, torch.optim.Optimizer),
        )


def get_recommended_hyper_parameter(
    dataset_name: str, model_name: str
) -> HyperParameter:
    """
    Given dataset and model, return a set of recommended hyper parameters
    """

    hyper_parameter = None

    if dataset_name == "MNIST" or (
        dataset_name == "FashionMNIST" and model_name.lower() == "LeNet5".lower()
    ):
        hyper_parameter = HyperParameter(
            epoch=50,
            batch_size=64,
            optimizer_kwargs={"learning_rate": 0.01, "fake_weight_decay": 1},
        )
    elif dataset_name == "CIFAR10":
        hyper_parameter = HyperParameter(
            epoch=350,
            batch_size=64,
            optimizer_kwargs={"learning_rate": 0.1, "fake_weight_decay": 1},
        )
    elif dataset_name == "CIFAR100":
        hyper_parameter = HyperParameter(
            epoch=350,
            batch_size=64,
            optimizer_kwargs={"fake_weight_decay": 1},
        )
    elif dataset_name == "SVHN":
        hyper_parameter = HyperParameter(
            epoch=50,
            batch_size=4,
            optimizer_kwargs={"learning_rate": 0.0001, "fake_weight_decay": 1},
        )
    else:
        hyper_parameter = HyperParameter()
    return hyper_parameter


@dataclass(kw_only=True)
class HyperParameterConfig(HyperParameter):
    weight_decay: None | float = None
    fake_weight_decay: None | float = None

    def create_hyper_parameter(self) -> HyperParameter:
        hyper_parameter = copy.copy(self)
        if self.weight_decay is not None:
            assert self.fake_weight_decay is None
            hyper_parameter.optimizer_kwargs["weight_decay"] = self.weight_decay
            hyper_parameter.optimizer_kwargs.pop("fake_weight_decay", None)
        else:
            if self.fake_weight_decay is not None:
                hyper_parameter.optimizer_kwargs["fake_weight_decay"] = (
                    self.fake_weight_decay
                )
            assert self.weight_decay is None
        return hyper_parameter


__all__ = [
    "lr_scheduler_step_after_batch",
    "HyperParameterAction",
    "HyperParameter",
    "HyperParameterConfig",
    "get_optimizer_names",
    "global_optimizer_factory",
]
