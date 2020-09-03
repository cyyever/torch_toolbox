import torch.nn as nn
import torch.optim as optim

from hyper_parameter import HyperParameter
from trainer import Trainer
from validator import Validator
from dataset import get_dataset, DatasetType
from models.lenet import LeNet5
from models.densenet2 import (
    densenet_CIFAR10,
    densenet_CIFAR10_group_norm,
)
from models.senet.se_resnet import se_resnet20
from models.senet.se_resnet_group_norm import se_resnet20_group_norm


def choose_loss_function(model):
    last_layer = list(model.modules())[-1]
    if isinstance(last_layer, nn.LogSoftmax):
        return nn.NLLLoss()
    if isinstance(last_layer, nn.Linear):
        return nn.CrossEntropyLoss()
    raise NotImplementedError()


def get_task_dataset_name(name):
    if name.startswith("CIFAR10_"):
        return "CIFAR10"
    if name.startswith("MNIST_"):
        return "MNIST"
    return name


def get_task_configuration(task_name: str, for_training: bool):
    model = None
    loss_fun = None
    hyper_parameter = None
    momentum = 0.9
    if task_name == "MNIST":
        model = LeNet5()
        if for_training:
            hyper_parameter = HyperParameter(
                epochs=50, batch_size=64, learning_rate=0.01, weight_decay=1
            )
            hyper_parameter.set_lr_scheduler_factory(
                lambda optimizer, _, __: optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=[5, 55, 555]
                )
            )

            # hyper_parameter.set_lr_scheduler_factory(
            #     lambda optimizer, hyper_parameter: optim.lr_scheduler.CosineAnnealingLR(
            #         optimizer, T_max=hyper_parameter.epochs))

    elif task_name == "FashionMNIST":
        model = LeNet5()
        if for_training:
            hyper_parameter = HyperParameter(
                epochs=50, batch_size=64, learning_rate=0.01, weight_decay=1
            )
            hyper_parameter.set_lr_scheduler_factory(
                lambda optimizer, _, __: optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, verbose=True, factor=0.5, patience=2
                )
            )
    elif task_name == "CIFAR10_se_resnet":
        model = se_resnet20_group_norm(num_classes=10)
        if for_training:
            hyper_parameter = HyperParameter(
                epochs=350, batch_size=128, learning_rate=0.1, weight_decay=5
            )
            hyper_parameter.set_lr_scheduler_factory(
                lambda optimizer,
                hyper_parameter,
                training_dataset_size: optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    pct_start=0.4,
                    max_lr=0.5,
                    total_steps=(
                        hyper_parameter.epochs *
                        (
                            (training_dataset_size +
                             hyper_parameter.batch_size -
                             1) //
                            hyper_parameter.batch_size)),
                    anneal_strategy="linear",
                    three_phase=True,
                    div_factor=10,
                ))
    elif task_name == "CIFAR10":
        model = densenet_CIFAR10()
        if for_training:
            hyper_parameter = HyperParameter(
                epochs=350, batch_size=128, learning_rate=0.1, weight_decay=1
            )

            hyper_parameter.set_lr_scheduler_factory(
                lambda optimizer, _, __: optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, verbose=True, factor=0.1
                )
            )
    elif task_name == "CIFAR10_densenet_group_norm":
        model = densenet_CIFAR10_group_norm()
        if for_training:
            hyper_parameter = HyperParameter(
                epochs=350, batch_size=128, learning_rate=0.1, weight_decay=1
            )
            # hyper_parameter.set_lr_scheduler_factory(
            #     lambda optimizer, _, __: optim.lr_scheduler.ReduceLROnPlateau(
            #         optimizer, verbose=True, factor=0.1
            #     )
            # )
            hyper_parameter.set_lr_scheduler_factory(
                lambda optimizer,
                hyper_parameter,
                training_dataset_size: optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=0.1,
                    total_steps=(
                        hyper_parameter.epochs *
                        (
                            (training_dataset_size +
                             hyper_parameter.batch_size -
                             1) //
                            hyper_parameter.batch_size)),
                    anneal_strategy="linear",
                    three_phase=True,
                    div_factor=10,
                ))
    else:
        raise NotImplementedError(task_name)

    dataset_name = get_task_dataset_name(task_name)

    if loss_fun is None:
        loss_fun = choose_loss_function(model)
    test_dataset = get_dataset(dataset_name, DatasetType.Test)
    if for_training:
        if hyper_parameter.optimizer_factory is None:
            hyper_parameter.set_optimizer_factory(
                lambda params, learning_rate, weight_decay, dataset: optim.SGD(
                    params,
                    momentum=momentum,
                    lr=learning_rate,
                    weight_decay=(weight_decay / len(dataset)),
                )
            )
        training_dataset = get_dataset(dataset_name, DatasetType.Training)
        trainer = Trainer(model, loss_fun, training_dataset, hyper_parameter)
        trainer.validation_dataset = get_dataset(
            dataset_name, DatasetType.Validation)
        trainer.test_dataset = test_dataset
        return trainer
    return Validator(model, loss_fun, test_dataset)
