import math
import torch.nn as nn
import torch.optim as optim

from .hyper_parameter import HyperParameter
from .trainer import Trainer
from .validator import Validator
from .dataset import get_dataset
from .model import LeNet5


def get_task_configuration(task_name, for_training, hyper_parameter=None):
    if task_name == "MNIST":
        training_dataset = get_dataset(task_name, True)
        validation_dataset = get_dataset(task_name, False)
        model = LeNet5()
        loss_fun = nn.CrossEntropyLoss()
        if for_training:
            trainer = Trainer(model, loss_fun, training_dataset, task_name)
            hyper_parameter = HyperParameter(
                epoches=50, batch_size=64, learning_rate=0.01
            )
            hyper_parameter.set_lr_scheduler_factory(
                lambda optimizer: optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=lambda epoch: math.pow(
                        0.5, math.floor(
                            epoch / 10)), ))

            trainer.set_hyper_parameter(hyper_parameter)
            trainer.set_validation_dataset(validation_dataset)
            return trainer
        validator = Validator(model, loss_fun, validation_dataset)
        return validator
    raise NotImplementedError(task_name)
