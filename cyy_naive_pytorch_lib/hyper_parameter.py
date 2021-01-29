import inspect
import multiprocessing
from typing import Callable, Optional

import torch
import torch.optim as optim
from cyy_naive_lib.log import get_logger

from dataset import dataset_with_indices
from ml_types import MachineLearningPhase


class HyperParameter:
    def __init__(
        self,
        epoch: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        momentum: float = 0.9,
    ):
        self.__epoch = epoch
        self.__batch_size = batch_size
        self.__learning_rate = learning_rate
        self.__weight_decay = weight_decay
        self.__momentum = momentum
        self.__collate_fn = None
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

    @property
    def learning_rate(self):
        return self.__learning_rate

    def set_learning_rate(self, learning_rate):
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

    def get_lr_scheduler(self, optimizer, training_dataset_size: int):
        assert self.__lr_scheduler_factory is not None
        return self.__lr_scheduler_factory(
            self, optimizer, training_dataset_size=training_dataset_size
        )

    @staticmethod
    def lr_scheduler_step_after_batch(lr_scheduler):
        return isinstance(lr_scheduler, torch.optim.lr_scheduler.OneCycleLR)

    @staticmethod
    def get_lr_scheduler_factory(name, dataset_name=None):
        def callback(hyper_parameter, optimizer, training_dataset_size):
            nonlocal dataset_name
            nonlocal name
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
                    max_lr=hyper_parameter.learning_rate * 5,
                    total_steps=(
                        hyper_parameter.epoch
                        * (
                            (training_dataset_size + hyper_parameter.batch_size - 1)
                            // hyper_parameter.batch_size
                        )
                    ),
                    anneal_strategy="linear",
                    three_phase=True,
                    div_factor=10,
                )
            raise RuntimeError("unknown learning rate scheduler:" + name)

        return callback

    def set_optimizer_factory(self, optimizer_factory: Callable):
        self.__optimizer_factory = optimizer_factory

    @staticmethod
    def get_optimizer_factory(name: str):
        if name == "SGD":
            return optim.SGD
        if name == "Adam":
            return optim.Adam
        raise RuntimeError("unknown optimizer:" + name)

    def get_optimizer(self, params, training_dataset_size: int):
        assert self.__optimizer_factory is not None
        kwargs: dict = {
            "params": params,
            "lr": self.learning_rate,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay / training_dataset_size,
        }

        sig = inspect.signature(self.__optimizer_factory)
        parameter_names = {
            p.name for p in sig.parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD
        }
        return self.__optimizer_factory(
            **{k: kwargs[k] for k in kwargs if k in parameter_names}
        )

    def set_dataloader_collate_fn(self, collate_fn):
        self.__collate_fn = collate_fn

    def get_dataloader(self, dataset, phase: MachineLearningPhase):
        return torch.utils.data.DataLoader(
            dataset_with_indices(dataset),
            batch_size=self.batch_size,
            shuffle=(phase == MachineLearningPhase.Training),
            collate_fn=self.__collate_fn,
            num_workers=multiprocessing.cpu_count(),
        )

    def __str__(self):
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
        if self.__optimizer_factory is not None:
            s += " optimizer:" + str(self.__optimizer_factory)
        # if self.lr_scheduler_factory is not None:
        #     s += str(self.lr_scheduler_factory)

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
    elif dataset_name == "WebankStreet":
        hyper_parameter = HyperParameter(
            epoch=50, batch_size=4, learning_rate=0.0001, weight_decay=1
        )
    elif dataset_name == "SVHN":
        hyper_parameter = HyperParameter(
            epoch=50, batch_size=4, learning_rate=0.0001, weight_decay=1
        )
    else:
        get_logger().error(
            "no hyper parameter for dataset %s and model %s", dataset_name, model_name
        )
        return None
    hyper_parameter.set_lr_scheduler_factory(
        HyperParameter.get_lr_scheduler_factory("ReduceLROnPlateau", dataset_name)
    )
    if model_name == "FasterRCNN":
        hyper_parameter.set_dataloader_collate_fn(
            lambda batch: (
                [d[0] for d in batch],
                [d[1] for d in batch],
                torch.Tensor([d[2] for d in batch]),
            )
        )
    hyper_parameter.set_optimizer_factory(HyperParameter.get_optimizer_factory("Adam"))
    return hyper_parameter
