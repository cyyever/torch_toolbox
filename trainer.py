import os

import copy
import torch

from .device import get_cpu_device
from .device import get_device
from .util import model_gradients_to_vector
from .validator import Validator
from .log import get_logger
from .visualization import Window


class Trainer:
    def __init__(self, model, loss_fun, training_dataset):
        self.name = model.__class__.__name__
        self.model = copy.deepcopy(model)
        self.loss_fun = loss_fun
        self.training_dataset = training_dataset
        self.validation_dataset = None
        self.hyper_parameter = None
        self.__reset_loss()

    def set_name(self, name):
        self.name = name

    def set_validation_dataset(self, validation_dataset):
        self.validation_dataset = validation_dataset

    def set_hyper_parameter(self, hyper_parameter):
        self.hyper_parameter = hyper_parameter

    def train(self, **callbacks):
        def pre_training_callback(trainer, optimizer, lr_scheduler):
            get_logger(
                trainer.name).info(
                "begin training,optimizer is %s ,lr_scheduler is %s, model is %s",
                optimizer,
                lr_scheduler,
                trainer.model,
            )

        callbacks = Trainer.__append_callback(
            callbacks, "pre_training_callback", pre_training_callback
        )

        def after_batch_callback(
            trainer, epoch, batch_index, batch_size, batch_loss, learning_rates
        ):
            if batch_index % (len(trainer.training_dataset) //
                              (10 * batch_size)) == 0:
                get_logger(
                    trainer.name).info(
                    "epoch: %s, batch: %s, learning rate: %s, batch training loss: %s",
                    epoch,
                    batch_index,
                    learning_rates,
                    batch_loss,
                )

        callbacks = Trainer.__append_callback(
            callbacks, "after_batch_callback", after_batch_callback
        )

        def after_epoch_callback(trainer, epoch, learning_rates):
            loss_win = Window.get("training & validation loss")
            get_logger(trainer.name).info(
                "epoch: %s, training loss: %s", epoch, trainer.training_loss[-1],
            )
            loss_win.plot_loss(epoch,
                               trainer.training_loss[-1],
                               "training loss")
            Window.get("learning rate").plot_learning_rate(
                epoch, learning_rates[0])
            if trainer.validation_dataset is None:
                return
            validation_epoch_interval = int(
                callbacks.get("validation_epoch_interval", 1)
            )
            if epoch % validation_epoch_interval == 0:
                validation_loss, accuracy = Validator(
                    trainer.model, trainer.loss_fun, trainer.validation_dataset
                ).validate(trainer.hyper_parameter.batch_size)
                validation_loss = validation_loss.data.item()
                trainer.validation_loss[epoch] = validation_loss
                get_logger(
                    trainer.name).info(
                    "epoch: %s, learning_rate: %s, validation loss: %s, accuracy = %s",
                    epoch,
                    learning_rates,
                    validation_loss,
                    accuracy,
                )
                loss_win.plot_loss(epoch, validation_loss, "validation loss")
                Window.get("validation accuracy").plot_accuracy(
                    epoch, accuracy, "accuracy"
                )

        callbacks = Trainer.__append_callback(
            callbacks, "after_epoch_callback", after_epoch_callback
        )

        return self.__train(**callbacks)

    def __train(self, **callbacks):
        optimizer = self.hyper_parameter.get_optimizer(self.model.parameters())
        lr_scheduler = self.hyper_parameter.get_lr_scheduler(optimizer)
        self.__reset_loss()
        training_data_loader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.hyper_parameter.batch_size,
            shuffle=True,
        )

        if "pre_training_callback" in callbacks:
            callbacks["pre_training_callback"](self, optimizer, lr_scheduler)

        training_set_size = len(self.training_dataset)
        batch_index = 0
        device = get_device()
        self.model.to(device)

        for epoch in range(self.hyper_parameter.epoches):
            self.model.train()
            training_loss = 0.0
            for batch in training_data_loader:
                self.model.to(device)
                optimizer.zero_grad()
                batch_loss = 0
                real_batch_size = batch[0].shape[0]

                cur_learning_rates = [group["lr"]
                                      for group in optimizer.param_groups]

                if "pre_batch_callback" in callbacks:
                    callbacks["pre_batch_callback"](
                        self.model, batch, batch_index, cur_learning_rates
                    )

                if "per_instance_gradient_callback" in callbacks:
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

                        if "per_instance_gradient_callback" in callbacks:
                            callbacks["per_instance_gradient_callback"](
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

                if hasattr(self.loss_fun, "reduction") and (
                    self.loss_fun.reduction == "mean"
                    or self.loss_fun.reduction == "elementwise_mean"
                ):
                    batch_loss *= real_batch_size
                    batch_loss /= training_set_size

                training_loss += batch_loss
                optimizer.step()
                if "after_batch_callback" in callbacks:
                    callbacks["after_batch_callback"](
                        self,
                        epoch,
                        batch_index,
                        real_batch_size,
                        batch_loss,
                        cur_learning_rates,
                    )

                batch_index += 1

            self.training_loss.append(training_loss)

            if "after_epoch_callback" in callbacks:
                callbacks["after_epoch_callback"](
                    self, epoch, cur_learning_rates)

            if isinstance(
                    lr_scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(self.training_loss[-1])
            else:
                lr_scheduler.step()

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

    def parameters(self, use_best_model=False):
        model = self.model
        if use_best_model:
            if self.min_training_loss_model:
                model = self.min_training_loss_model
            else:
                raise ValueError("no min model to use")

        model.to(get_cpu_device())
        return model.parameters()

    def __reset_loss(self):
        self.min_training_loss = None
        self.min_training_loss_model = None
        self.training_loss = []
        self.validation_loss = {}

    @staticmethod
    def __append_callback(callbacks, name, new_fun):
        old_callback = callbacks.get(name, None)

        def new_callback(*args, **kwargs):
            nonlocal old_callback
            if old_callback is not None:
                old_callback(*args, **kwargs)
            new_fun(*args, **kwargs)

        callbacks[name] = new_callback
        return callbacks
