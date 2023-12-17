import copy
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any

import torch
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.reflection import call_fun, get_class_attrs

from ..data_structure.torch_thread_task_queue import TorchThreadTaskQueue
from .lr_finder import LRFinder


def _determine_learning_rate(task: Any, **kwargs: Any) -> float:
    tmp_trainer = task
    tmp_trainer.disable_stripable_hooks()
    lr_finder = LRFinder()
    get_logger().warning("register lr_finder %s", id(tmp_trainer))
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
    epoch: int = 350
    batch_size: int = 64
    learning_rate: HyperParameterAction | float = HyperParameterAction.FIND_LR
    learning_rate_scheduler_name: str = "ReduceLROnPlateau"
    learning_rate_scheduler_kwargs: dict = field(default_factory=lambda: {})
    optimizer_name: str = "Adam"
    optimizer_kwargs: dict = field(
        default_factory=lambda: {"momentum": 0.9, "fake_weight_decay": 1.0}
    )

    def __get_learning_rate(self, trainer: Any | None = None) -> float:
        if isinstance(self.learning_rate, HyperParameterAction):
            assert trainer is not None
            task_queue = TorchThreadTaskQueue()
            task_queue.start(worker_fun=_determine_learning_rate)
            trainer.offload_from_device()
            task_queue.add_task(copy.deepcopy(trainer))
            data = task_queue.get_data()
            assert data is not None
            learning_rate = data[0]
            assert isinstance(learning_rate, float)
            self.learning_rate = learning_rate
            task_queue.stop()
        return self.learning_rate

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

    @staticmethod
    def get_optimizer_names() -> list[str]:
        return sorted(HyperParameter.__get_optimizer_classes().keys())

    @staticmethod
    def get_lr_scheduler_names() -> list[str]:
        return sorted(HyperParameter.__get_learning_rate_scheduler_classes().keys())

    def get_optimizer(self, trainer: Any, parameters=None) -> Any:

        optimizer_class = self.__get_optimizer_classes().get(self.optimizer_name, None)
        if optimizer_class is None:
            raise RuntimeError(
                f"unknown optimizer:{self.optimizer_name}, supported names are: {list(self.__get_optimizer_classes().keys())}"
            )

        kwargs = copy.copy(self.optimizer_kwargs)
        if parameters is None:
            parameters = list(trainer.model.parameters())
        kwargs |= {
            "params": parameters,
            "lr": self.__get_learning_rate(trainer=trainer),
        }
        if "foreach" not in kwargs:
            kwargs["foreach"] = not torch.backends.mps.is_available()
        kwargs.pop("learning_rate", None)
        if "fake_weight_decay" in kwargs:
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
        hyper_parameter = HyperParameter()
    return hyper_parameter


@dataclass(kw_only=True)
class HyperParameterConfig(HyperParameter):
    weight_decay: None | float = 1.0
    fake_weight_decay: None | float = 1.0

    def create_hyper_parameter(self) -> HyperParameter:
        hyper_parameter = copy.copy(self)
        if self.fake_weight_decay is not None:
            hyper_parameter.optimizer_kwargs[
                "fake_weight_decay"
            ] = self.fake_weight_decay
        if self.weight_decay is not None:
            hyper_parameter.optimizer_kwargs["fake_weight_decay"] = self.weight_decay
        return hyper_parameter
        # get_recommended_hyper_parameter(dataset_name, model_name)
