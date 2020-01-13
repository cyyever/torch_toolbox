import copy
import torch
import torch.nn.functional as F

import gradient as gd
from .device import get_device


class validator:
    def __init__(
        self, model, loss_fun, validation_dataset,
    ):
        self.model = copy.deepcopy(model)
        self.loss_fun = loss_fun
        self.validation_dataset = validation_dataset

    def validate(self, batch_size, after_batch_callback=None):
        validation_data_loader = torch.utils.data.DataLoader(
            self.validation_dataset, batch_size=batch_size
        )
        num_correct = 0
        num_examples = 0
        device = get_device()
        self.model.eval()
        self.model.to(device)
        validation_loss = torch.zeros(1)
        validation_loss = validation_loss.to(device)
        for batch in validation_data_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = self.model(inputs)
            batch_loss = self.loss_fun(outputs, targets)
            if self.loss_fun.reduction == "mean":
                batch_loss *= len(outputs)
            if after_batch_callback:
                after_batch_callback(self.model, batch_loss)
            validation_loss += batch_loss
            correct = torch.eq(
                torch.max(F.softmax(outputs, dim=1), dim=1)[1], targets
            ).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        validation_loss /= len(self.validation_dataset)
        return (validation_loss, num_correct / num_examples)

    def get_gradient(self):
        gradient = None

        def after_batch_callback(model, batch_loss):
            nonlocal gradient
            if gradient is None:
                gradient = gd.get_gradient(model, batch_loss)
            else:
                gradient += gd.get_gradient(model, batch_loss)

        self.validate(64, after_batch_callback)
        return gradient / len(self.validation_dataset)
