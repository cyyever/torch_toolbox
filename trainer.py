import os
import pickle
import copy
import torch
import torch.optim as optim

from .device import get_cpu_device
from .device import get_device
from .util import model_gradients_to_vector
from .validator import Validator
from .log import get_logger


class Trainer:
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
        self.lr_scheduler_fun = None

    def set_optimizer_function(self, optimizer_fun):
        self.optimizer_fun = optimizer_fun

    def set_lr_scheduler(self, lr_scheduler_fun):
        self.lr_scheduler_fun = lr_scheduler_fun

    def train(self, epochs, batch_size, init_learning_rate, **kwargs):
        training_data_loader = torch.utils.data.DataLoader(
            self.training_dataset, batch_size=batch_size, shuffle=True
        )

        device = get_device()
        self.model.to(device)
        optimizer = self.optimizer_fun(
            self.model.parameters(),
            lr=init_learning_rate)
        lr_scheduler = None

        if self.lr_scheduler_fun is not None:
            lr_scheduler = self.lr_scheduler_fun(optimizer)
        else:
            lr_scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=(lambda epoch: 1)
            )

        instance_size = len(self.training_dataset)

        batch_index = 0
        for epoch in range(epochs):
            self.model.train()
            training_loss = 0.0
            for batch in training_data_loader:
                self.model.to(device)
                optimizer.zero_grad()
                batch_loss = 0
                real_batch_size = batch[0].shape[0]
                cur_learning_rates = lr_scheduler.get_last_lr()

                if "pre_batch_callback" in kwargs:
                    kwargs["pre_batch_callback"](
                        self.model, batch, batch_index, cur_learning_rates
                    )

                if "per_instance_gradient_callback" in kwargs:
                    prev_accumulated_gradient = None
                    instance_inputs, instance_targets, instance_indices = batch
                    for i, instance_index in enumerate(instance_indices):
                        instance_index = instance_indices[i].data.item()
                        instance_input = instance_inputs[i].to(device)
                        instance_target = instance_targets[i].to(device)
                        output = self.model(torch.stack([instance_input]))
                        loss = self.loss_fun(
                            output, torch.stack(
                                [instance_target]))
                        batch_loss += loss.data.item() / real_batch_size
                        loss.backward()
                        cur_accumulated_gradient = model_gradients_to_vector(
                            self.model)
                        instance_gradient = None
                        if prev_accumulated_gradient is None:
                            instance_gradient = cur_accumulated_gradient
                        else:
                            instance_gradient = (
                                cur_accumulated_gradient - prev_accumulated_gradient)
                        prev_accumulated_gradient = cur_accumulated_gradient

                        if "per_instance_gradient_callback" in kwargs:
                            kwargs["per_instance_gradient_callback"](
                                self.model,
                                instance_index,
                                instance_gradient,
                                cur_learning_rates,
                                real_batch_size,
                            )
                else:
                    inputs = batch[0].to(device)
                    targets = batch[1].to(device)
                    outputs = self.model(inputs)
                    loss = self.loss_fun(outputs, targets)
                    batch_loss = loss.data.item()
                    loss.backward()

                get_logger().info(
                    "trainer: %s, epoch: %s, batch: %s, learning rate: %s, batch training loss: %s",
                    self.name,
                    epoch,
                    batch_index,
                    cur_learning_rates,
                    batch_loss,
                )
                if "after_batch_callback" in kwargs:
                    kwargs["after_batch_callback"](
                        self.model, batch, batch_index, cur_learning_rates
                    )

                optimizer.step()
                batch_index += 1

                if hasattr(self.loss_fun, "reduction") and (
                    self.loss_fun.reduction == "mean"
                    or self.loss_fun.reduction == "elementwise_mean"
                ):
                    batch_loss *= real_batch_size
                    batch_loss /= instance_size
                training_loss += batch_loss

            get_logger().info(
                "trainer:%s, epoch: %s, epoch training loss: %s",
                self.name,
                epoch,
                training_loss,
            )

            if "validation_dataset" in kwargs:
                validation_epoch_interval = int(
                    kwargs.get("validation_epoch_interval", 1)
                )
                assert validation_epoch_interval > 0

                if epoch % validation_epoch_interval == 0:
                    validation_loss, accuracy = Validator(
                        self.model, self.loss_fun, kwargs["validation_dataset"]
                    ).validate(batch_size)
                    get_logger().info(
                        "trainer:%s, epoch: %s, validation loss: %s, accuracy = %s",
                        self.name,
                        epoch,
                        validation_loss.data.item(),
                        accuracy,
                    )

            lr_scheduler.step()
            if "after_epoch_callback" in kwargs:
                kwargs["after_epoch_callback"](self.model, epoch)
            # if self.min_training_loss is None or training_loss < self.min_training_loss:
            #     self.min_training_loss = training_loss
            #     self.min_training_loss_model = copy.deepcopy(self.model)
            #     self.min_training_loss_model.to(get_cpu_device())

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
