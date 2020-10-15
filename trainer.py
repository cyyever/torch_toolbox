import os
import datetime
import copy
import torch

from cyy_naive_lib.log import get_logger
from cyy_naive_lib.list_op import split_list_to_chunks

from device import get_device
from model_util import ModelUtil
from validator import Validator
from per_sample_gradient import get_per_sample_gradient
from visualization import EpochWindow, Window


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

    def __init__(self, model, loss_fun, training_dataset, hyper_parameter):
        self.model = copy.deepcopy(model)
        self.loss_fun = loss_fun
        self.training_dataset = training_dataset
        self.stop_criterion = None
        self.validation_dataset = None
        self.test_dataset = None
        self.__hyper_parameter = hyper_parameter
        self.__reset_hyper_parameter = False
        self.__visdom_env = None
        self.__reset_loss()

    def get_validator(self, use_test_data=True):
        if use_test_data:
            return Validator(self.model, self.loss_fun, self.test_dataset)
        return Validator(self.model, self.loss_fun, self.validation_dataset)

    def set_hyper_parameter(self, hyper_parameter):
        self.__hyper_parameter = hyper_parameter
        self.__reset_hyper_parameter = True

    def get_hyper_parameter(self):
        return self.__hyper_parameter

    def load_model(self, model_path):
        self.model = torch.load(model_path, map_location=get_device())


    def save_model(self, save_dir, model_name="model.pt"):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model, os.path.join(save_dir, model_name))

    def repeated_train(self, repeated_num, save_dir, **kwargs):
        def training_callback(_, trainer):
            nonlocal save_dir, kwargs
            trainer.train(**kwargs)
            trainer.save_model(save_dir, with_timestamp=True)
            return {
                "training_loss": trainer.training_loss,
                "validation_loss": trainer.validation_loss,
                "validation_accuracy": trainer.validation_accuracy,
            }

        return Trainer.repeated_training(repeated_num, self, training_callback)

    def train(self, **kwargs):
        self.__visdom_env = (
            "training_"
            + str(self.model.__class__.__name__)
            + "_{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now())
        )

        def pre_training_callback(trainer, optimizer, lr_scheduler):
            model_util = ModelUtil(trainer.model)
            get_logger().info(
                "begin training for %s epochs,hyper_parameter is %s,optimizer is %s ,lr_scheduler is %s, model is %s, loss function is %s, parameter number is %s",
                self.__hyper_parameter.epochs,
                self.__hyper_parameter,
                optimizer,
                lr_scheduler,
                trainer.model.__class__.__name__,
                trainer.loss_fun,
                len(model_util.get_parameter_list()),
            )

        kwargs = Trainer.__prepend_callback(
            kwargs, "pre_training_callbacks", pre_training_callback
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
            kwargs, "after_batch_callbacks", after_batch_callback
        )

        plot_parameter_distribution = kwargs.get(
            "plot_parameter_distribution", False)
        plot_class_accuracy = kwargs.get("plot_class_accuracy", False)

        def plot_loss_after_epoch(trainer, epoch, learning_rates, **kwargs):
            nonlocal plot_parameter_distribution
            nonlocal plot_class_accuracy
            if plot_parameter_distribution:
                layer_win = Window(
                    "parameter distribution",
                    env=trainer.__visdom_env)

                layer_win.plot_histogram(
                    ModelUtil(trainer.model).get_parameter_list())

            EpochWindow(
                "learning rate",
                env=trainer.__visdom_env).plot_learning_rate(
                epoch,
                learning_rates[0])
            optimizer = kwargs.get("optimizer", None)
            momentums = [group["momentum"] for group in optimizer.param_groups]
            EpochWindow("momentum", env=trainer.__visdom_env).plot_scalar(
                epoch, momentums[0], y_label="Momentum"
            )

            loss_win = EpochWindow(
                "training & validation loss", env=trainer.__visdom_env
            )
            get_logger().info(
                "epoch: %s, training loss: %s",
                epoch,
                trainer.training_loss[-1],
            )
            loss_win.plot_loss(epoch,
                               trainer.training_loss[-1],
                               "training loss")

            validation_epoch_interval = int(
                kwargs.get("validation_epoch_interval", 1))
            if (
                trainer.validation_dataset is not None
                and epoch % validation_epoch_interval == 0
            ):
                (validation_loss, accuracy, other_data,) = trainer.get_validator(
                    use_test_data=False
                ).validate(
                    trainer.__hyper_parameter.batch_size, per_class_accuracy=True
                )
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
                EpochWindow(
                    "validation accuracy", env=trainer.__visdom_env
                ).plot_accuracy(epoch, accuracy, "accuracy")

                if plot_class_accuracy:
                    class_accuracy = other_data["per_class_accuracy"]
                    for idx, sub_list in enumerate(
                        split_list_to_chunks(list(class_accuracy.keys()), 2)
                    ):
                        class_accuracy_win = EpochWindow(
                            "class accuracy part " + str(idx), env=trainer.__visdom_env)
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

            test_epoch_interval = int(kwargs.get("test_epoch_interval", 5))
            if trainer.test_dataset is not None and (
                epoch % test_epoch_interval == 0
                or epoch == trainer.get_hyper_parameter().epochs
            ):
                (test_loss, accuracy, other_data,) = trainer.get_validator(
                    use_test_data=True
                ).validate(
                    trainer.__hyper_parameter.batch_size, per_class_accuracy=False
                )
                test_loss = test_loss.data.item()
                trainer.test_loss[epoch] = test_loss
                trainer.test_accuracy[epoch] = accuracy
                EpochWindow(
                    "test accuracy",
                    env=trainer.__visdom_env).plot_accuracy(
                    epoch,
                    accuracy,
                    "accuracy")
                get_logger().info(
                    "epoch: %s, learning_rate: %s, test loss: %s, accuracy = %s",
                    epoch,
                    learning_rates,
                    test_loss,
                    accuracy,
                )
                Window.save_envs()

        kwargs = Trainer.__prepend_callback(
            kwargs, "after_epoch_callbacks", plot_loss_after_epoch
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
        lr_scheduler = self.__hyper_parameter.get_lr_scheduler(
            optimizer, training_set_size
        )
        lr_step_after_batch = False
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.OneCycleLR):
            lr_step_after_batch = True
            get_logger().info("adjust lr after batch")
        for callback in kwargs.get("pre_training_callbacks", []):
            callback(self, optimizer, lr_scheduler)

        for epoch in range(1, self.__hyper_parameter.epochs + 1):
            if self.__reset_hyper_parameter:
                self.__reset_hyper_parameter = False
                optimizer = self.__hyper_parameter.get_optimizer(
                    self.model.parameters(), self.training_dataset
                )
                lr_scheduler = self.__hyper_parameter.get_lr_scheduler(
                    optimizer, training_set_size
                )
                get_logger().warning("use new hyper-parameter")

            training_loss = 0.0
            cur_learning_rates = [group["lr"]
                                  for group in optimizer.param_groups]
            for batch in training_data_loader:
                if lr_step_after_batch:
                    cur_learning_rates = [
                        group["lr"] for group in optimizer.param_groups
                    ]
                self.model.train()
                self.model.to(device)
                optimizer.zero_grad()
                real_batch_size = batch[0].shape[0]

                for callback in kwargs.get("pre_batch_callbacks", []):
                    callback(self, batch, batch_index)

                instance_inputs = batch[0].to(device)
                instance_targets = batch[1].to(device)
                instance_indices = None
                if len(batch) >= 3:
                    instance_indices = [idx.data.item() for idx in batch[2]]

                if "per_sample_gradient_callback" in kwargs:
                    assert instance_indices is not None
                    per_sample_gradient_callback, computed_indices = kwargs[
                        "per_sample_gradient_callback"
                    ]
                    sample_gradient_inputs = []
                    sample_gradient_targets = []
                    sample_gradient_indices = []
                    for (
                            instance_input,
                            instance_target,
                            instance_index) in zip(
                            instance_inputs,
                            instance_targets,
                            instance_indices):
                        if (
                            computed_indices is not None
                            and instance_index not in computed_indices
                        ):
                            continue
                        sample_gradient_inputs.append(instance_input)
                        sample_gradient_targets.append(instance_target)
                        sample_gradient_indices.append(instance_index)
                    if sample_gradient_indices:
                        gradient_list = get_per_sample_gradient(
                            self.model,
                            self.loss_fun,
                            sample_gradient_inputs,
                            sample_gradient_targets,
                        )

                        assert len(gradient_list) == len(
                            sample_gradient_indices)
                        for (sample_gradient, index) in zip(
                            gradient_list, sample_gradient_indices
                        ):
                            per_sample_gradient_callback(
                                self,
                                index,
                                sample_gradient,
                                optimizer=optimizer,
                            )
                optimizer.zero_grad()
                outputs = self.model(instance_inputs)
                loss = self.loss_fun(outputs, instance_targets)
                batch_loss = loss.data.item()
                loss.backward()

                normalized_batch_loss = batch_loss
                if hasattr(self.loss_fun, "reduction") and (
                    self.loss_fun.reduction == "mean"
                    or self.loss_fun.reduction == "elementwise_mean"
                ):
                    normalized_batch_loss *= real_batch_size
                    normalized_batch_loss /= training_set_size

                training_loss += normalized_batch_loss

                for callback in kwargs.get("after_batch_callbacks", []):
                    callback(
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
                if lr_step_after_batch:
                    lr_scheduler.step()
                batch_index += 1

            self.training_loss.append(training_loss)
            for callback in kwargs.get("after_epoch_callbacks", []):
                callback(self, epoch, cur_learning_rates, optimizer=optimizer)

            if self.stop_criterion is not None and self.stop_criterion(
                self, epoch, cur_learning_rates
            ):
                get_logger().warning("early stop")
                break

            if not lr_step_after_batch:
                if isinstance(lr_scheduler,
                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(
                        self.training_loss[-1] + self.validation_loss[epoch]
                    )
                else:
                    lr_scheduler.step()

    def __reset_loss(self):
        self.min_training_loss = None
        self.min_training_loss_model = None
        self.training_loss = []
        self.validation_loss = {}
        self.validation_accuracy = {}
        self.test_loss = {}
        self.test_accuracy = {}

    @staticmethod
    def __prepend_callback(kwargs, name, new_fun):
        callbacks = kwargs.get(name, [])
        callbacks.insert(0, new_fun)
        kwargs[name] = callbacks
        return kwargs
