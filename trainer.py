import os
import pickle
import copy
import torch

from .device import get_cpu_device
from .device import get_device
from .util import model_gradients_to_vector
from .validator import Validator
from .log import get_logger


class Trainer:
    def __init__(self, model, loss_fun, training_dataset):
        self.name = model.__class__.__name__
        self.model = copy.deepcopy(model)
        self.loss_fun = loss_fun
        self.training_dataset = training_dataset
        self.validation_dataset = None
        self.hyper_parameter = None
        self.min_training_loss = None
        self.min_training_loss_model = None

    def set_name(self, name):
        self.name = name

    def set_validation_dataset(self, validation_dataset):
        self.validation_dataset = validation_dataset

    def set_hyper_parameter(self, hyper_parameter):
        self.hyper_parameter = hyper_parameter

    def train(self, **kwargs):
        optimizer = self.hyper_parameter.get_optimizer(self.model.parameters())
        lr_scheduler = self.hyper_parameter.get_lr_scheduler(optimizer)

        get_logger(
            self.name).info(
            "begin training,lr_scheduler is %s",
            lr_scheduler)
        get_logger(self.name).info("begin training,optimizer is %s", optimizer)

        training_data_loader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.hyper_parameter.batch_size,
            shuffle=True,
        )
        device = get_device()
        self.model.to(device)

        instance_size = len(self.training_dataset)
        batch_index = 0
        for epoch in range(self.hyper_parameter.epoches):
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

                if batch_index % (1000 // real_batch_size) == 0:
                    get_logger(
                        self.name).info(
                        "epoch: %s, batch: %s, learning rate: %s, batch training loss: %s",
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

            get_logger(self.name).info(
                "epoch: %s, epoch training loss: %s", epoch, training_loss,
            )

            if "validation_dataset" in kwargs:
                validation_epoch_interval = int(
                    kwargs.get("validation_epoch_interval", 1)
                )
                assert validation_epoch_interval > 0

                if (
                    epoch % validation_epoch_interval == 0
                    and self.validation_dataset is not None
                ):
                    validation_loss, accuracy = Validator(
                        self.model, self.loss_fun, self.validation_dataset
                    ).validate(self.hyper_parameter.batch_size)
                    get_logger(
                        self.name).info(
                        "epoch: %s, learning_rate:%s, validation loss: %s, accuracy = %s",
                        epoch,
                        cur_learning_rates,
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
