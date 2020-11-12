from typing import Callable, Optional
import torch.optim as optim
from cyy_naive_lib.log import get_logger


class HyperParameter:
    def __init__(
        self,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        momentum: float = 0.9,
    ):
        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__learning_rate = learning_rate
        self.__weight_decay = weight_decay
        self.__momentum = momentum
        self.__lr_scheduler_factory: Optional[Callable] = None
        self.__optimizer_factory: Optional[Callable] = None

    @property
    def epochs(self):
        return self.__epochs

    def set_epochs(self, epochs):
        self.__epochs = epochs

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
            optimizer, self, training_dataset_size)

    def set_optimizer_factory(self, optimizer_factory: Callable):
        self.__optimizer_factory = optimizer_factory

    def get_optimizer(self, params, training_dataset_size: int):
        assert self.__optimizer_factory is not None
        kwargs: dict = {
            "params": params,
            "lr": self.epochs,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay / training_dataset_size,
        }
        return self.__optimizer_factory(**kwargs)

    def __str__(self):
        s = (
            "epochs:"
            + str(self.epochs)
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


def get_default_lr_scheduler(
    optimizer, hyper_parameter: HyperParameter, training_dataset_size: int
):
    return optim.lr_scheduler.OneCycleLR(
        optimizer,
        pct_start=0.4,
        max_lr=0.5,
        total_steps=(
            hyper_parameter.epochs
            * (
                (training_dataset_size + hyper_parameter.batch_size - 1)
                // hyper_parameter.batch_size
            )
        ),
        anneal_strategy="linear",
        three_phase=True,
        div_factor=10,
    )


def get_optimizer_factory(name: str):
    if name == "SGD":
        return optim.SGD
    raise RuntimeError("unknown optimizer:" + name)


def get_recommended_hyper_parameter(
    dataset_name: str, model_name: str
) -> Optional[HyperParameter]:
    """
    Given dataset and model, return a set of recommended hyper parameters
    """

    hyper_parameter = None
    if dataset_name == "MNIST":
        hyper_parameter = HyperParameter(
            epochs=50, batch_size=64, learning_rate=0.01, weight_decay=1
        )
        hyper_parameter.set_lr_scheduler_factory(get_default_lr_scheduler)
    elif dataset_name == "FashionMNIST" and model_name.lower() == "LeNet5".lower():
        hyper_parameter = HyperParameter(
            epochs=50, batch_size=64, learning_rate=0.01, weight_decay=1
        )
        hyper_parameter.set_lr_scheduler_factory(
            lambda optimizer, _, __: optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, verbose=True, factor=0.5, patience=2
            )
        )
    elif dataset_name == "CIFAR10":
        hyper_parameter = HyperParameter(
            epochs=350, batch_size=128, learning_rate=0.1, weight_decay=1
        )

        hyper_parameter.set_lr_scheduler_factory(
            lambda optimizer, _, __: optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, verbose=True, factor=0.1
            )
        )
    else:
        get_logger().error(
            "no hyper parameter for dataset %s and model %s",
            dataset_name,
            model_name)
        return None
    hyper_parameter.set_optimizer_factory(get_optimizer_factory("SGD"))
    return hyper_parameter
