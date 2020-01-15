import os
import pickle
import copy
import torch
import torch.optim as optim

from .device import get_cpu_device
from .device import get_device


class trainer:
    def __init__(
        self, model, loss_funs, training_datasets, name="",
    ):
        self.model = copy.deepcopy(model)
        if not isinstance(loss_funs, list):
            loss_funs = [loss_funs]
        if not isinstance(training_datasets, list):
            training_datasets = [training_datasets]
        if len(loss_funs) != len(training_datasets):
            raise ValueError(
                "loss_funs shape does not match training_datasets shape")
        self.loss_funs = loss_funs
        self.training_datasets = training_datasets
        self.name = name
        self.min_training_loss = None
        self.min_training_loss_model = None

    def train(
        self, epochs, batch_size, learning_rate, after_training_callback=None,
    ):
        training_data_loaders = [
            torch.utils.data.DataLoader(training_dataset, batch_size=batch_size)
            for training_dataset in self.training_datasets
        ]

        device = get_device()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            self.model.train()
            training_loss = 0.0
            for training_data_loader, loss_fun in zip(
                training_data_loaders, self.loss_funs
            ):
                for batch in training_data_loader:
                    optimizer.zero_grad()
                    inputs, targets = batch
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = self.model(inputs)
                    loss = loss_fun(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    batch_loss = loss.data.item()
                    if hasattr(loss_fun, "reduction") and (
                        loss_fun.reduction == "mean"
                        or loss_fun.reduction == "elementwise_mean"
                    ):
                        batch_loss *= len(outputs)
                    training_loss += batch_loss
                if hasattr(loss_fun, "reduction") and (
                    loss_fun.reduction == "mean"
                    or loss_fun.reduction == "elementwise_mean"
                ):
                    training_loss /= len(training_data_loader.dataset)
            print(
                "trainer:{}, epoch: {}, training loss: {}".format(
                    self.name, epoch, training_loss
                )
            )
            if self.min_training_loss is None or training_loss < self.min_training_loss:
                self.min_training_loss = training_loss
                self.min_training_loss_model = copy.deepcopy(self.model)

            if after_training_callback:
                after_training_callback(epoch, self.model)

    def save(self, save_dir, save_min_model=False):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model = self.model
        if save_min_model:
            if self.min_training_loss_model:
                model = self.min_training_loss_model
            else:
                raise ValueError("no min model to save")

        torch.save(model, os.path.join(save_dir, "model.pt"))

    def save_dataset(self, save_dir):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        open(os.path.join(save_dir, "training_dataset"), "wb").write(
            pickle.dumps(self.training_datasets)
        )

    def parameters(self):
        self.model.to(get_cpu_device())
        return self.model.parameters()
