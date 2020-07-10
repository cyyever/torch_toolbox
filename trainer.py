import os
import copy
import datetime
import torch

from cyy_naive_lib.log import get_logger

from device import get_cpu_device, get_device
from util import (
    model_parameters_to_vector,
    split_list_to_chunks,
)
from validator import Validator
from visualization import EpochWindow, Window
from gradient import get_gradient


class Trainer:
    @staticmethod
    def repeated_training(number, trainer, training_callback):
        results = dict()
        for idx in range(number):
            statistics = training_callback(idx, trainer)
            assert isinstance(statistics, dict)
            for k, v in statistics.items():
                tensor = None
                if isinstance(v, list):
                    tensor = torch.Tensor(v)
                elif isinstance(v, dict):
                    tensor = torch.Tensor([v[k] for k in sorted(v.keys())])
                else:
                    raise RuntimeError("unsupported value" + str(v))
                if k in results:
                    results[k] += tensor
                else:
                    results[k] = tensor
        for k, v in results.items():
            results[k] = v / number
        return results

    def __init__(self, model, loss_fun, training_dataset):
        self.model = copy.deepcopy(model)
        self.loss_fun = loss_fun
        self.training_dataset = training_dataset
        self.validation_dataset = None
        self.__hyper_parameter = None
        self.__reset_hyper_parameter = False
        self.stop_criterion = None
        self.__reset_loss()

    def set_hyper_parameter(self, hyper_parameter):
        self.__hyper_parameter = hyper_parameter
        self.__reset_hyper_parameter = True

    def get_hyper_parameter(self):
        return self.__hyper_parameter

    def repeated_train(self, repeated_num, save_dir, **kwargs):
        def training_callback(idx, trainer):
            trainer.train(**kwargs)
            trainer.save(save_dir, with_timestamp=True)
            return {
                "training_loss": trainer.training_loss,
                "validation_loss": trainer.validation_loss,
                "validation_accuracy": trainer.validation_accuracy,
            }

        Trainer.repeated_training(repeated_num, self, training_callback)

    def train(self, **kwargs):
        def pre_training_callback(trainer, optimizer, lr_scheduler):
            get_logger().info(
                "begin training for %s epochs,hyper_parameter is %s,optimizer is %s ,lr_scheduler is %s, model is %s",
                self.__hyper_parameter.epochs,
                self.__hyper_parameter,
                optimizer,
                lr_scheduler,
                trainer.model,
            )

        kwargs = Trainer.__prepend_callback(
            kwargs, "pre_training_callback", pre_training_callback
        )

        def after_batch_callback(
            trainer,
            epoch,
            batch_index,
            batch_size,
            batch_loss,
            learning_rates,
            **kwargs
        ):
            ten_batches = len(trainer.training_dataset) // (10 * batch_size)
            if ten_batches == 0 or batch_index % ten_batches == 0:
                get_logger().info(
                    "epoch: %s, batch: %s, learning rate: %s, batch training loss: %s",
                    epoch,
                    batch_index,
                    learning_rates,
                    batch_loss,
                )

        kwargs = Trainer.__prepend_callback(
            kwargs, "after_batch_callback", after_batch_callback
        )

        plot_parameter_distribution = kwargs.get(
            "plot_parameter_distribution", False)
        plot_class_accuracy = kwargs.get("plot_class_accuracy", False)

        def after_epoch_callback(trainer, epoch, learning_rates):
            nonlocal plot_parameter_distribution
            nonlocal plot_class_accuracy
            if plot_parameter_distribution:
                layer_win = Window("parameter distribution")

                layer_win.plot_histogram(
                    model_parameters_to_vector(
                        trainer.model))

            loss_win = EpochWindow("training & validation loss")
            get_logger().info("epoch: %s, training loss: %s",
                              epoch, trainer.training_loss[-1], )
            loss_win.plot_loss(epoch,
                               trainer.training_loss[-1],
                               "training loss")
            EpochWindow("learning rate").plot_learning_rate(
                epoch, learning_rates[0])
            if trainer.validation_dataset is None:
                return
            validation_epoch_interval = int(
                kwargs.get("validation_epoch_interval", 1))
            if epoch % validation_epoch_interval == 0:
                validation_loss, accuracy, other_data = Validator(
                    trainer.model, trainer.loss_fun, trainer.validation_dataset).validate(
                    trainer.__hyper_parameter.batch_size, per_class_accuracy=True)
                validation_loss = validation_loss.data.item()
                trainer.validation_loss[epoch] = validation_loss
                trainer.validation_accuracy[epoch] = accuracy
                get_logger().info(
                    "epoch: %s, learning_rate: %s, validation loss: %s, accuracy = %s",
                    epoch,
                    learning_rates,
                    validation_loss,
                    accuracy,
                )
                loss_win.plot_loss(epoch, validation_loss, "validation loss")
                EpochWindow("validation accuracy").plot_accuracy(
                    epoch, accuracy, "accuracy"
                )

                if plot_class_accuracy:
                    class_accuracy = other_data["per_class_accuracy"]
                    for idx, sub_list in enumerate(
                        split_list_to_chunks(list(class_accuracy.keys()), 2)
                    ):
                        class_accuracy_win = EpochWindow(
                            "class accuracy part " + str(idx)
                        )
                        for k in sub_list:
                            get_logger().info(
                                "epoch: %s, learning_rate: %s, class %s accuracy = %s",
                                epoch,
                                learning_rates,
                                k,
                                class_accuracy[k],
                            )
                            class_accuracy_win.plot_accuracy(
                                epoch,
                                class_accuracy[k],
                                "class_" + str(k) + "_accuracy",
                            )

        kwargs = Trainer.__prepend_callback(
            kwargs, "after_epoch_callback", after_epoch_callback
        )

        return self.__train(**kwargs)

    def __train(self, **kwargs):
        training_data_loader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.__hyper_parameter.batch_size,
            shuffle=True,
        )

        training_set_size = len(self.training_dataset)
        get_logger().info("training_set_size is %s", training_set_size)
        batch_index = 0
        device = get_device()
        get_logger().info("use device %s", device)
        self.model.to(device)
        self.__reset_hyper_parameter = False
        self.__reset_loss()
        optimizer = self.__hyper_parameter.get_optimizer(
            self.model.parameters(), self.training_dataset
        )
        lr_scheduler = self.__hyper_parameter.get_lr_scheduler(optimizer)
        if "pre_training_callback" in kwargs:
            kwargs["pre_training_callback"](self, optimizer, lr_scheduler)

        for epoch in range(1, self.__hyper_parameter.epochs + 1):
            if self.__reset_hyper_parameter:
                self.__reset_hyper_parameter = False
                optimizer = self.__hyper_parameter.get_optimizer(
                    self.model.parameters(), self.training_dataset
                )
                lr_scheduler = self.__hyper_parameter.get_lr_scheduler(
                    optimizer)
                get_logger().warning("use new hyper-parameter")

            training_loss = 0.0
            cur_learning_rates = [group["lr"]
                                  for group in optimizer.param_groups]
            for batch in training_data_loader:
                self.model.train()
                self.model.to(device)
                optimizer.zero_grad()
                real_batch_size = batch[0].shape[0]

                if "pre_batch_callback" in kwargs:
                    kwargs["pre_batch_callback"](self, batch, batch_index)

                instance_inputs = batch[0].to(device)
                instance_targets = batch[1].to(device)
                instance_indices = None
                if len(batch) >= 3:
                    instance_indices = [idx.data.item() for idx in batch[2]]

                if "per_instance_gradient_callback" in kwargs:
                    per_instance_gradient_callback, computed_indices = kwargs[
                        "per_instance_gradient_callback"
                    ]
                    for i, instance_index in enumerate(instance_indices):
                        if (
                            computed_indices is not None
                            and instance_index not in computed_indices
                        ):
                            continue
                        instance_input = instance_inputs[i]
                        instance_target = instance_targets[i]
                        output = self.model(torch.stack([instance_input]))
                        loss = self.loss_fun(
                            output, torch.stack(
                                [instance_target]))

                        instance_gradient = get_gradient(self.model, loss)
                        per_instance_gradient_callback(
                            self,
                            instance_index,
                            instance_gradient,
                            optimizer=optimizer,
                        )
                optimizer.zero_grad()
                outputs = self.model(instance_inputs)
                loss = self.loss_fun(outputs, instance_targets)
                batch_loss = loss.data.item()
                loss.backward()

                if hasattr(self.loss_fun, "reduction") and (
                    self.loss_fun.reduction == "mean"
                    or self.loss_fun.reduction == "elementwise_mean"
                ):
                    batch_loss *= real_batch_size
                    batch_loss /= training_set_size

                training_loss += batch_loss

                if "after_batch_callback" in kwargs:
                    kwargs["after_batch_callback"](
                        self,
                        epoch,
                        batch_index,
                        real_batch_size,
                        batch_loss,
                        cur_learning_rates,
                        instance_indices=instance_indices,
                        optimizer=optimizer,
                    )
                optimizer.step()
                cur_learning_rates = [group["lr"]
                                      for group in optimizer.param_groups]
                batch_index += 1

            self.training_loss.append(training_loss)

            if "after_epoch_callback" in kwargs:
                kwargs["after_epoch_callback"](self, epoch, cur_learning_rates)

            if self.stop_criterion is not None and self.stop_criterion(
                self, epoch, cur_learning_rates
            ):
                get_logger().warning("early stop")
                break

            if isinstance(
                    lr_scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(
                    self.training_loss[-1] + self.validation_loss[epoch])
            else:
                lr_scheduler.step()

    def save(self, save_dir, with_timestamp=False):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model = self.model
        name = "model.pt"
        if with_timestamp:
            name = "model_{date:%Y_%m_%d_%H_%M_%S}.pt".format(
                date=datetime.datetime.now()
            )

        torch.save(model, os.path.join(save_dir, name))

    def parameters(self):
        model = self.model
        model.to(get_cpu_device())
        return model.parameters()

    def __reset_loss(self):
        self.min_training_loss = None
        self.min_training_loss_model = None
        self.training_loss = []
        self.validation_loss = {}
        self.validation_accuracy = {}

    @staticmethod
    def __prepend_callback(kwargs, name, new_fun):
        old_callback = kwargs.get(name, None)

        def new_callback(*args, **kwargs):
            new_fun(*args, **kwargs)
            nonlocal old_callback
            if old_callback is not None:
                old_callback(*args, **kwargs)

        kwargs[name] = new_callback
        return kwargs
