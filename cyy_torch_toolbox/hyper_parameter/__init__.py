import copy
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any, Callable

import torch
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.reflection import call_fun, get_class_attrs

from ..data_structure.torch_thread_task_queue import TorchThreadTaskQueue
from .lr_finder import LRFinder


def _determine_learning_rate(task: Any, **kwargs: Any) -> float:
    tmp_trainer = task
    tmp_trainer.disable_stripable_hooks()
    lr_finder = LRFinder()
    get_logger().warning("register lr_finder")
    tmp_trainer.prepend_hook(lr_finder)
    tmp_trainer.train()
    get_logger().warning(
        "suggested_learning_rate is %s", lr_finder.suggested_learning_rate
    )
    assert lr_finder.suggested_learning_rate is not None
    return lr_finder.suggested_learning_rate


def lr_scheduler_step_after_batch(
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> bool:
    return isinstance(lr_scheduler, torch.optim.lr_scheduler.OneCycleLR)


class HyperParameterAction(StrEnum):
    FIND_LR = auto()


@dataclass(kw_only=True)
class HyperParameter:
    epoch: int
    batch_size: int = 8
    learning_rate_scheduler_name: str = "ReduceLROnPlateau"
    learning_rate_scheduler_kwargs: dict = field(default_factory=lambda: {})
    _optimizer_factory: None | Callable = None
    optimizer_kwargs: dict = field(default_factory=lambda: {})

    def __get_learning_rate(self, trainer: Any | None = None) -> float:
        if isinstance(
            self.optimizer_kwargs.get("learning_rate", HyperParameterAction.FIND_LR),
            HyperParameterAction,
        ):
            assert trainer is not None
            task_queue = TorchThreadTaskQueue()
            task_queue.start(worker_fun=_determine_learning_rate)
            trainer.offload_from_device()
            task_queue.add_task(copy.deepcopy(trainer))
            data = task_queue.get_data()
            assert data is not None
            learning_rate = data[0]
            assert isinstance(learning_rate, float)
            self.optimizer_kwargs["learning_rate"] = learning_rate
            task_queue.stop()
        return self.optimizer_kwargs["learning_rate"]

    def get_iterations_per_epoch(self, dataset_size: int) -> int:
        if self.batch_size == 1:
            return dataset_size
        return (dataset_size + self.batch_size - 1) // self.batch_size

    def get_lr_scheduler(self, trainer) -> torch.optim.lr_scheduler.LRScheduler:
        return self.__get_lr_scheduler_factory(
            trainer=trainer, name=self.learning_rate_scheduler_name
        )

    def __get_lr_scheduler_factory(
        self, trainer: Any, name: str
    ) -> torch.optim.lr_scheduler.LRScheduler:
        optimizer = trainer.get_optimizer()
        training_dataset_size = trainer.dataset_size
        full_kwargs: dict = {}
        full_kwargs["optimizer"] = optimizer
        if name == "ReduceLROnPlateau":
            patience = min(10, self.epoch + 9 // 10)
            full_kwargs["patience"] = patience
            full_kwargs["factor"] = 0.1
            full_kwargs["verbose"] = True
            full_kwargs.update(self.learning_rate_scheduler_kwargs)
            get_logger().debug(
                "ReduceLROnPlateau patience is %s", full_kwargs["patience"]
            )
            return torch.optim.lr_scheduler.ReduceLROnPlateau(**full_kwargs)
        if name == "OneCycleLR":
            full_kwargs["pct_start"] = 0.4
            full_kwargs["max_lr"] = 10 * self.__get_learning_rate(trainer)
            full_kwargs["total_steps"] = self.epoch * self.get_iterations_per_epoch(
                training_dataset_size
            )
            full_kwargs["anneal_strategy"] = "linear"
            full_kwargs["three_phase"] = True
            full_kwargs.update(self.learning_rate_scheduler_kwargs)
            return torch.optim.lr_scheduler.OneCycleLR(**full_kwargs)
        if name == "CosineAnnealingLR":
            full_kwargs["T_max"] = self.epoch
            full_kwargs.update(self.learning_rate_scheduler_kwargs)
            return torch.optim.lr_scheduler.CosineAnnealingLR(**full_kwargs)
        if name == "MultiStepLR":
            full_kwargs["T_max"] = self.epoch
            full_kwargs["milestones"] = [30, 80]
            full_kwargs.update(self.learning_rate_scheduler_kwargs)
            return torch.optim.lr_scheduler.MultiStepLR(**full_kwargs)
        fun = getattr(torch.optim.lr_scheduler, name)
        if fun is not None:
            full_kwargs.update(self.learning_rate_scheduler_kwargs)
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
    def get_optimizer_names() -> list[str]:
        return sorted(HyperParameter.__get_optimizer_classes().keys())

    @staticmethod
    def get_lr_scheduler_names() -> list[str]:
        return [
            "LambdaLR",
            "MultiplicativeLR",
            "StepLR",
            "MultiStepLR",
            "ConstantLR",
            "LinearLR",
            "ExponentialLR",
            "SequentialLR",
            "CosineAnnealingLR",
            "ChainedScheduler",
            "ReduceLROnPlateau",
            "CyclicLR",
            "CosineAnnealingWarmRestarts",
            "OneCycleLR",
            "PolynomialLR",
            "LRScheduler",
        ]

    def get_optimizer(self, trainer: Any, parameters=None) -> Any:
        assert self._optimizer_factory is not None
        foreach = not torch.backends.mps.is_available()
        kwargs = copy.copy(self.optimizer_kwargs)
        if parameters is None:
            parameters = trainer.model.parameters()
        kwargs |= {
            "params": parameters,
            "lr": self.__get_learning_rate(trainer=trainer),
            "foreach": foreach,
        }
        kwargs.pop("learning_rate", None)
        if "fake_weight_decay" in kwargs:
            kwargs["weight_decay"] = (
                kwargs.pop("fake_weight_decay") / trainer.dataset_size
            )
        return call_fun(self._optimizer_factory, kwargs)

    @staticmethod
    def __get_optimizer_classes() -> dict:
        return get_class_attrs(
            torch.optim,
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
        hyper_parameter = HyperParameter(
            epoch=350,
            batch_size=64,
            optimizer_kwargs={"fake_weight_decay": 0},
        )
    hyper_parameter.learning_rate_scheduler_name = "ReduceLROnPlateau"
    hyper_parameter.set_optimizer_factory("Adam")
    return hyper_parameter


@dataclass(kw_only=True)
class HyperParameterConfig:
    epoch: int = 350
    batch_size: int = 64
    learning_rate: HyperParameterAction | float = HyperParameterAction.FIND_LR
    momentum: None | float = 0.9
    weight_decay: None | float = None
    fake_weight_decay: None | float = None
    learning_rate_scheduler_name: str = "ReduceLROnPlateau"
    learning_rate_scheduler_kwargs: dict = field(default_factory=lambda: {})
    optimizer_name: str = "Adam"
    optimizer_kwargs: dict = field(default_factory=lambda: {})

    def create_hyper_parameter(self) -> HyperParameter:
        hyper_parameter = HyperParameter(epoch=self.epoch, batch_size=self.batch_size)

        hyper_parameter.optimizer_kwargs["learning_rate"] = self.learning_rate
        if self.momentum is not None:
            hyper_parameter.optimizer_kwargs["momentum"] = self.momentum
        if self.fake_weight_decay is not None:
            hyper_parameter.optimizer_kwargs[
                "fake_weight_decay"
            ] = self.fake_weight_decay
        if self.weight_decay is not None:
            hyper_parameter.optimizer_kwargs["fake_weight_decay"] = self.weight_decay
        hyper_parameter.set_optimizer_factory(self.optimizer_name)
        hyper_parameter.learning_rate_scheduler_name = self.learning_rate_scheduler_name
        hyper_parameter.learning_rate_scheduler_kwargs = (
            self.learning_rate_scheduler_kwargs
        )
        return hyper_parameter
        # get_recommended_hyper_parameter(dataset_name, model_name)
