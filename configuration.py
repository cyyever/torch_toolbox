import torch.nn as nn
import torch.optim as optim
import torchvision

from hyper_parameter import HyperParameter
from trainer import Trainer
from validator import Validator
from dataset import get_dataset
from models.lenet import LeNet5
from models.densenet import densenet_cifar


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
    return name


def get_task_configuration(task_name, for_training):
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
                lambda optimizer: optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=[5, 55, 555]
                )
            )

    elif task_name == "FashionMNIST":
        model = LeNet5()
        if for_training:
            hyper_parameter = HyperParameter(
                epochs=50, batch_size=64, learning_rate=0.01, weight_decay=1
            )

            hyper_parameter.set_lr_scheduler_factory(
                lambda optimizer: optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, verbose=True, factor=0.5, patience=2
                )
            )

    elif task_name == "CIFAR10":
        model = densenet_cifar()
        if for_training:
            hyper_parameter = HyperParameter(
                epochs=350, batch_size=128, learning_rate=0.1, weight_decay=1
            )

            hyper_parameter.set_lr_scheduler_factory(
                lambda optimizer: optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, verbose=True, factor=0.1
                )
            )
            # hyper_parameter.set_lr_scheduler_factory(
            #     lambda optimizer: optim.lr_scheduler.StepLR(
            #         optimizer, step_size=10))
    elif task_name == "CIFAR10_LENET":
        model = LeNet5(input_channels=3)
        if for_training:
            hyper_parameter = HyperParameter(
                epochs=350, batch_size=128, learning_rate=0.1, weight_decay=1
            )

            hyper_parameter.set_lr_scheduler_factory(
                lambda optimizer: optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, verbose=True, factor=0.1
                )
            )

    elif task_name == "STL10":
        model = torchvision.models.densenet121(num_classes=10)
        if for_training:
            hyper_parameter = HyperParameter(
                epochs=350, batch_size=32, learning_rate=0.1
            )

            hyper_parameter.set_lr_scheduler_factory(
                lambda optimizer: optim.lr_scheduler.StepLR(
                    optimizer, step_size=50))

    else:
        raise NotImplementedError(task_name)

    dataset_name = get_task_dataset_name(task_name)
    validation_dataset = get_dataset(dataset_name, False)

    if loss_fun is None:
        loss_fun = choose_loss_function(model)
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
        training_dataset = get_dataset(dataset_name, True)
        trainer = Trainer(model, loss_fun, training_dataset)
        trainer.set_hyper_parameter(hyper_parameter)
        trainer.validation_dataset = validation_dataset
        return trainer
    validator = Validator(model, loss_fun, validation_dataset)
    return validator
