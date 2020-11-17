import os
import datetime
import copy
from typing import Callable, Optional

import torch

from cyy_naive_lib.log import get_logger
from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks

from algorithm.per_sample_gradient import get_per_sample_gradient
from device import get_device, put_data_to_device
from model_util import ModelUtil
from util import get_batch_size
from inference import Inferencer
from model_loss import ModelWithLoss
from visualization import EpochWindow, Window
from hyper_parameter import HyperParameter
from phase import MachineLearningPhase


class BasicTrainer:
    def __init__(
        self,
        model_with_loss: ModelWithLoss,
        training_dataset,
        hyper_parameter: Optional[HyperParameter],
    ):
        self.__model_with_loss = copy.deepcopy(model_with_loss)
        self.__training_dataset = training_dataset
        self.__validation_dataset: Optional[torch.utils.data.Dataset] = None
        self.__test_dataset: Optional[torch.utils.data.Dataset] = None
        self.__stop_criterion: Optional[Callable] = None
        self.__hyper_parameter = hyper_parameter
        self.__reset_hyper_parameter = False
        self.__visdom_env = None
        self.__reset_loss()

    @property
    def model_with_loss(self):
        return self.__model_with_loss

    @property
    def model(self):
        return self.model_with_loss.model

    @property
    def loss_fun(self):
        return self.model_with_loss.loss_fun

    @property
    def training_dataset(self):
        return self.__training_dataset

    def set_training_dataset(self, training_dataset: torch.utils.data.Dataset):
        self.__training_dataset = training_dataset

    @property
    def validation_dataset(self):
        return self.__validation_dataset

    def set_validation_dataset(
            self, validation_dataset: torch.utils.data.Dataset):
        self.__validation_dataset = validation_dataset

    @property
    def test_dataset(self):
        return self.__test_dataset

    def set_test_dataset(self, test_dataset: torch.utils.data.Dataset):
        self.__test_dataset = test_dataset

    def get_inferencer(self):
        return Inferencer(
            self.model_with_loss,
            self.test_dataset,
            self.hyper_parameter)

    def set_hyper_parameter(self, hyper_parameter):
        self.__hyper_parameter = hyper_parameter
        self.__reset_hyper_parameter = True

    @property
    def hyper_parameter(self):
        return self.__hyper_parameter

    def load_model(self, model_path):
        self.model_with_loss.set_model(
            torch.load(model_path, map_location=get_device())
        )

    def save_model(self, save_dir, model_name="model.pt"):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model, os.path.join(save_dir, model_name))

    def repeated_train(self, repeated_num, save_dir=None, **kwargs):
        def training_callback(_, trainer: BasicTrainer):
            nonlocal save_dir, kwargs
            trainer.train(**kwargs)
            if save_dir is not None:
                trainer.save_model(save_dir)
            return {
                "training_loss": trainer.training_loss,
                "validation_loss": trainer.validation_loss,
                "validation_accuracy": trainer.validation_accuracy,
            }

        return BasicTrainer.__repeated_training(
            repeated_num, self, training_callback)

    def train(self, **kwargs):
        assert self.hyper_parameter is not None

        training_set_size = len(self.training_dataset)
        get_logger().info("training_set_size is %s", training_set_size)
        device = get_device()
        get_logger().info("use device %s", device)
        self.model.to(device)
        self.__reset_hyper_parameter = True
        self.__reset_loss()
        optimizer = None
        lr_scheduler = None
        lr_step_after_batch = None

        for epoch in range(1, self.__hyper_parameter.epochs + 1):
            if self.__reset_hyper_parameter:
                self.__reset_hyper_parameter = False
                optimizer = self.__hyper_parameter.get_optimizer(
                    self.model.parameters(), len(self.training_dataset)
                )
                lr_scheduler = self.__hyper_parameter.get_lr_scheduler(
                    optimizer, training_set_size
                )
                if epoch != 1:
                    get_logger().warning("use new hyper-parameter")
                lr_step_after_batch = False
                if isinstance(lr_scheduler,
                              torch.optim.lr_scheduler.OneCycleLR):
                    lr_step_after_batch = True
                    get_logger().info("adjust lr after batch")
            if epoch == 1:
                for callback in kwargs.get("pre_training_callbacks", []):
                    callback(self, optimizer, lr_scheduler)
            training_loss = 0.0
            cur_learning_rates = [group["lr"]
                                  for group in optimizer.param_groups]
            for batch_index, batch in enumerate(
                self.__hyper_parameter.get_dataloader(
                    self.training_dataset, phase=MachineLearningPhase.Training
                )
            ):
                if lr_step_after_batch:
                    cur_learning_rates = [
                        group["lr"] for group in optimizer.param_groups
                    ]
                self.model.train()
                self.model.to(device)
                optimizer.zero_grad()

                for callback in kwargs.get("pre_batch_callbacks", []):
                    callback(self, batch, batch_index)

                instance_inputs = put_data_to_device(batch[0], device)
                instance_targets = put_data_to_device(batch[1], device)
                instance_indices = None
                if len(batch) >= 3:
                    instance_indices = [idx.data.item() for idx in batch[2]]

                real_batch_size = get_batch_size(instance_inputs)

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
                            self.model_with_loss,
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
                result = self.model_with_loss(
                    instance_inputs,
                    instance_targets,
                    phase=MachineLearningPhase.Training,
                )
                loss = result["loss"]
                batch_loss = loss.data.item()
                loss.backward()

                normalized_batch_loss = batch_loss
                if self.model_with_loss.is_averaged_loss():
                    normalized_batch_loss *= real_batch_size
                    normalized_batch_loss /= training_set_size

                training_loss += normalized_batch_loss
                optimizer.step()
                if lr_step_after_batch:
                    lr_scheduler.step()

                for callback in kwargs.get("after_batch_callbacks", []):
                    callback(
                        self,
                        batch_index,
                        epoch=epoch,
                        cur_learning_rates=cur_learning_rates,
                        batch_loss=batch_loss,
                        cur_batch_size=real_batch_size,
                        instance_indices=instance_indices,
                        optimizer=optimizer,
                        training_set_size=training_set_size,
                    )

            self.training_loss.append(training_loss)
            for callback in kwargs.get("after_epoch_callbacks", []):
                callback(self, epoch, cur_learning_rates, optimizer=optimizer)

            if self.__stop_criterion is not None and self.__stop_criterion(
                self, epoch, cur_learning_rates
            ):
                get_logger().warning("early stop")
                break

            if not lr_step_after_batch:
                if isinstance(lr_scheduler,
                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                    get_logger().debug(
                        "call ReduceLROnPlateau for total loss %s",
                        self.training_loss[-1] + self.validation_loss[epoch],
                    )
                    lr_scheduler.step(
                        self.training_loss[-1] + self.validation_loss[epoch]
                    )
                else:
                    lr_scheduler.step()

    def __reset_loss(self):
        self.training_loss = []
        self.validation_loss = {}
        self.validation_accuracy = {}
        self.test_loss = {}
        self.test_accuracy = {}

    @staticmethod
    def __repeated_training(number: int, trainer, training_callback: Callable):
        results: dict = dict()
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

    @staticmethod
    def prepend_callback(kwargs, name, new_fun):
        callbacks = kwargs.get(name, [])
        callbacks.insert(0, new_fun)
        kwargs[name] = callbacks
        return kwargs


class Trainer(BasicTrainer):
    def train(self, **kwargs):
        self.__visdom_env = (
            "training_"
            + str(self.model.__class__.__name__)
            + "_"
            + str(self.training_dataset)
            + "_{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now())
        )

        def pre_training_callback(trainer, optimizer, lr_scheduler):
            model_util = ModelUtil(trainer.model)
            get_logger().info(
                "begin training, hyper_parameter is %s, optimizer is %s ,lr_scheduler is %s, %s, parameter number is %s",
                trainer.hyper_parameter,
                optimizer,
                lr_scheduler,
                trainer.model_with_loss,
                len(model_util.get_parameter_list()),
            )

        kwargs = BasicTrainer.prepend_callback(
            kwargs, "pre_training_callbacks", pre_training_callback
        )

        def after_batch_callback(_, batch_index, **kwargs):
            training_set_size = kwargs["training_set_size"]
            ten_batches = training_set_size // (10 * kwargs["cur_batch_size"])
            if ten_batches == 0 or batch_index % ten_batches == 0:
                get_logger().info(
                    "epoch: %s, batch: %s, learning rate: %s, batch training loss: %s",
                    kwargs["epoch"],
                    batch_index,
                    kwargs["cur_learning_rates"],
                    kwargs["batch_loss"],
                )

        kwargs = BasicTrainer.prepend_callback(
            kwargs, "after_batch_callbacks", after_batch_callback
        )

        plot_parameter_distribution = kwargs.get(
            "plot_parameter_distribution", False)
        plot_class_accuracy = kwargs.get("plot_class_accuracy", False)

        def plot_loss_after_epoch(
            trainer: BasicTrainer, epoch, learning_rates, **kwargs
        ):
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
                (
                    validation_loss,
                    accuracy,
                    other_data,
                ) = trainer.get_inferencer().inference(per_class_accuracy=True)
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
                or epoch == trainer.hyper_parameter.epochs
            ):
                (test_loss, accuracy, other_data, ) = trainer.get_inferencer(
                ).inference(per_class_accuracy=False)
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

        kwargs = BasicTrainer.prepend_callback(
            kwargs, "after_epoch_callbacks", plot_loss_after_epoch
        )
        return super().train(**kwargs)
