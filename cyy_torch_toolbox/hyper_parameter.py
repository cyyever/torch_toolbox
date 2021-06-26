import copy
import inspect
from enum import IntEnum, auto
from typing import Callable, Optional, Union

import torch
import torch.optim as optim
from cyy_naive_lib.log import get_logger

from algorithm.lr_finder import LRFinder


class HyperParameterAction(IntEnum):
    FIND_LR = auto()


class HyperParameter:
    def __init__(
        self,
        epoch: int,
        batch_size: int,
        learning_rate: Union[float, HyperParameterAction],
        weight_decay: float,
        momentum: float = 0.9,
    ):
        self.__epoch = epoch
        self.__batch_size = batch_size
        self.__learning_rate = learning_rate
        self.__weight_decay = weight_decay
        self.__momentum = momentum
        # self.__collate_fn = None
        self.__lr_scheduler_factory: Optional[Callable] = None
        self.__optimizer_factory: Optional[Callable] = None

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
            get_logger().warning("guess lr")
            tmp_trainer = copy.deepcopy(trainer)
            tmp_trainer.disable_stripable_hooks()
            lr_finder = LRFinder()
            tmp_trainer.prepend_hook(lr_finder)
            tmp_trainer.train()
            get_logger().warning(
                "suggested_learning_rate is %s", lr_finder.suggested_learning_rate
            )
            self.__learning_rate = lr_finder.suggested_learning_rate
            # self.__learning_rate = max(suggesstion_lrs.lr_min), suggesstion_lrs.lr_steep)
        return self.__learning_rate

    def set_learning_rate(self, learning_rate: Union[float, HyperParameterAction]):
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

    def set_lr_scheduler_factory(self, lr_scheduler_factory: Callable):
        self.__lr_scheduler_factory = lr_scheduler_factory

    def get_iterations_per_epoch(self, training_dataset_size):
        if self.batch_size == 1:
            return training_dataset_size
        return (training_dataset_size + self.batch_size - 1) // self.batch_size

    def get_lr_scheduler(self, trainer):
        assert self.__lr_scheduler_factory is not None
        return self.__lr_scheduler_factory(self, trainer)

    @staticmethod
    def lr_scheduler_step_after_batch(lr_scheduler):
        return isinstance(lr_scheduler, torch.optim.lr_scheduler.OneCycleLR)

    @staticmethod
    def get_lr_scheduler_factory(name, dataset_name=None):
        def hook(hyper_parameter, trainer):
            nonlocal dataset_name
            nonlocal name
            optimizer = trainer.get_optimizer()
            training_dataset_size = len(trainer.dataset)
            if name == "ReduceLROnPlateau":
                patience = min(10, hyper_parameter.epoch + 9 // 10)
                if dataset_name == "CIFAR10":
                    patience = 2
                get_logger().info("ReduceLROnPlateau patience is %s", patience)
                return optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    verbose=True,
                    factor=0.1,
                    patience=patience,
                )
            if name == "OneCycleLR":
                return optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    pct_start=0.4,
                    max_lr=10 * hyper_parameter.get_learning_rate(trainer),
                    total_steps=hyper_parameter.epoch
                    * hyper_parameter.get_iterations_per_epoch(training_dataset_size),
                    anneal_strategy="linear",
                    three_phase=True,
                )
            if name == "CosineAnnealingLR":
                return optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=hyper_parameter.epoch
                )
            if name == "MultiStepLR":
                return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80])

            raise RuntimeError("unknown learning rate scheduler:" + name)

        return hook

    def set_optimizer_factory(self, optimizer_factory: Callable):
        self.__optimizer_factory = optimizer_factory

    @staticmethod
    def get_optimizer_factory(name: str):
        if name == "SGD":
            return optim.SGD
        if name == "Adam":
            return optim.Adam
        raise RuntimeError("unknown optimizer:" + name)

    def get_optimizer(self, trainer):
        assert self.__optimizer_factory is not None
        kwargs: dict = {
            "params": trainer.model.parameters(),
            "lr": self.get_learning_rate(trainer),
            "momentum": self.momentum,
            "weight_decay": self.weight_decay / len(trainer.dataset),
        }

        sig = inspect.signature(self.__optimizer_factory)
        parameter_names = {
            p.name for p in sig.parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD
        }
        return self.__optimizer_factory(
            **{k: kwargs[k] for k in kwargs if k in parameter_names}
        )

    # def set_dataloader_collate_fn(self, collate_fn):
    #     self.__collate_fn = collate_fn

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
    # if model_name == "FasterRCNN":
    #     hyper_parameter.set_dataloader_collate_fn(
    #         lambda batch: (
    #             [d[0] for d in batch],
    #             [d[1] for d in batch],
    #             torch.Tensor([d[2] for d in batch]),
    #         )
    #     )
    hyper_parameter.set_optimizer_factory(HyperParameter.get_optimizer_factory("SGD"))
    return hyper_parameter


class HyperParameterConfig:
    def __init__(self):
        self.epoch = None
        self.batch_size = None
        self.find_learning_rate = True
        self.learning_rate = None
        self.learning_rate_scheduler = None
        self.momentum = None
        self.weight_decay = None
        self.optimizer_name = None

    def add_args(self, parser):
        parser.add_argument("--epoch", type=int, default=None)
        parser.add_argument("--batch_size", type=int, default=None)
        parser.add_argument("--learning_rate", type=float, default=None)
        parser.add_argument("--learning_rate_scheduler", type=str, default=None)
        parser.add_argument("--find_learning_rate", action="store_true", default=False)
        parser.add_argument("--momentum", type=float, default=None)
        parser.add_argument("--weight_decay", type=float, default=None)
        parser.add_argument("--optimizer_name", type=str, default=None)

    def load_args(self, args):
        for attr in dir(args):
            if attr.startswith("_"):
                continue
            if not hasattr(self, attr):
                continue
            get_logger().debug("set dataset collection config attr %s", attr)
            value = getattr(args, attr)
            if value is not None:
                setattr(self, attr, value)

    def create_hyper_parameter(self, dataset_name, model_name):
        hyper_parameter = get_recommended_hyper_parameter(dataset_name, model_name)

        if self.epoch is not None:
            hyper_parameter.set_epoch(self.epoch)
        if self.batch_size is not None:
            hyper_parameter.set_batch_size(self.batch_size)
        if self.learning_rate is not None and self.find_learning_rate:
            raise RuntimeError(
                "can't not specify a learning_rate and find a learning_rate at the same time"
            )
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
                HyperParameter.get_lr_scheduler_factory(self.learning_rate_scheduler)
            )
        return hyper_parameter
