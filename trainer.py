import os
import pickle
import copy
import torch
import torch.optim as optim

from .device import get_cpu_device
from .device import get_device
from .util import model_gradients_to_vector


class trainer:
    def __init__(
        self, model, loss_fun, training_dataset, name="",
    ):
        self.model = copy.deepcopy(model)
        self.loss_fun = loss_fun
        self.training_dataset = training_dataset
        self.name = name
        self.min_training_loss = None
        self.min_training_loss_model = None
        self.optimizer_fun = optim.Adam

    def set_optimizer_function(self, optimizer_fun):
        self.optimizer_fun = optimizer_fun

    def train(self, epochs, batch_size, learning_rate, **kwargs):
        training_data_loader = torch.utils.data.DataLoader(
            self.training_dataset, batch_size=batch_size, shuffle=True
        )

        device = get_device()
        self.model.to(device)
        optimizer = self.optimizer_fun(
            self.model.parameters(), lr=learning_rate)
        example_size = len(self.training_dataset)

        for epoch in range(epochs):
            self.model.train()
            training_loss = 0.0
            batch_index = 0
            for batch in training_data_loader:
                if "pre_batch_callback" in kwargs:
                    kwargs["pre_batch_callback"](
                        self.model, batch, learning_rate)
                self.model.to(device)
                optimizer.zero_grad()
                batch_loss = 0
                if "per_example_gradient_callback" in kwargs:
                    prev_accumulated_gradient = None
                    example_inputs, example_targets, example_indices = batch
                    real_batch_size = len(example_indices)
                    for i, example_index in enumerate(example_indices):
                        example_index = example_indices[i]
                        example_input = example_inputs[i].to(device)
                        example_target = example_targets[i].to(device)
                        output = self.model(torch.stack([example_input]))
                        loss = self.loss_fun(
                            output, torch.stack([example_target]))
                        batch_loss += loss.data.item() / real_batch_size
                        loss.backward()
                        cur_accumulated_gradient = model_gradients_to_vector(
                            self.model)
                        example_gradient = None
                        if prev_accumulated_gradient is None:
                            example_gradient = cur_accumulated_gradient
                        else:
                            example_gradient = cur_accumulated_gradient - prev_accumulated_gradient

                        if "per_example_gradient_callback" in kwargs:
                            kwargs["per_example_gradient_callback"](
                                self.model, example_index, example_gradient, learning_rate)
                else:
                    inputs = batch[0].to(device)
                    targets = batch[1].to(device)
                    outputs = self.model(inputs)
                    loss = self.loss_fun(outputs, targets)
                    batch_loss = loss.data.item()
                    loss.backward()

                optimizer.step()
                print(
                    "trainer:{}, epoch: {}, batch: {}, batch training loss: {}".format(
                        self.name, epoch, batch_index, batch_loss))
                batch_index += 1
                training_loss += (batch_loss * real_batch_size / example_size)
                if "after_batch_callback" in kwargs:
                    kwargs["after_batch_callback"](
                        self.model, batch, learning_rate)

            print(
                "trainer:{}, epoch: {}, epoch training loss: {}".format(
                    self.name, epoch, training_loss
                )
            )
            if "after_epoch_callback" in kwargs:
                kwargs["after_epoch_callback"](self.model, epoch)
            if self.min_training_loss is None or training_loss < self.min_training_loss:
                self.min_training_loss = training_loss
                self.min_training_loss_model = copy.deepcopy(self.model)
                self.min_training_loss_model.to(get_cpu_device())

    def save(self, save_dir, save_min_model=False):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model = self.model
        if save_min_model:
            if self.min_training_loss_model:
                model = self.min_training_loss_model
            else:
                raise ValueError("no min model to save")
        model.to(get_cpu_device())
        torch.save(model, os.path.join(save_dir, "model.pt"))

    def save_dataset(self, save_dir):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        open(os.path.join(save_dir, "training_dataset"), "wb").write(
            pickle.dumps(self.training_dataset)
        )

    def parameters(self, use_best_model=False):
        model = self.model
        if use_best_model:
            if self.min_training_loss_model:
                model = self.min_training_loss_model
            else:
                raise ValueError("no min model to use")

        model.to(get_cpu_device())
        return model.parameters()
