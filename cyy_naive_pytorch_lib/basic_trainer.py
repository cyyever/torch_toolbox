import copy
import logging
import os
from typing import Callable, List, Optional

import torch
from cyy_naive_lib.log import get_logger

from device import get_device, put_data_to_device
from hyper_parameter import HyperParameter
from inference import ClassificationInferencer, DetectionInferencer, Inferencer
from ml_types import MachineLearningPhase, ModelType
from model_loss import ModelWithLoss
from tensor import get_batch_size


class BasicTrainer:
    def __init__(
        self,
        model_with_loss: ModelWithLoss,
        training_dataset,
        hyper_parameter: HyperParameter,
    ):
        self.__model_with_loss = copy.deepcopy(model_with_loss)
        self.__training_dataset = training_dataset
        self.__validation_dataset: Optional[torch.utils.data.Dataset] = None
        self.__test_dataset: Optional[torch.utils.data.Dataset] = None
        self.__hyper_parameter = hyper_parameter
        self.__device = get_device()
        self.__data: dict = dict()
        self.__callbacks: dict[str, List[Callable]] = dict()
        self.__clear_loss()
        self.add_callback(
            "pre_batch_callbacks",
            lambda trainer, batch, batch_index: self.set_data(
                "cur_learning_rates",
                [group["lr"] for group in trainer.get_optimizer().param_groups],
            ),
        )

    @property
    def model_with_loss(self):
        return self.__model_with_loss

    @property
    def model(self) -> torch.nn.Module:
        return self.model_with_loss.model

    def get_data(self, key: str):
        return self.__data.get(key)

    def set_data(self, key: str, value):
        self.__data[key] = value

    def get_callbacks(self, name: str) -> List[Callable]:
        return self.__callbacks.get(name, [])

    def add_callback(self, name: str, cb: Callable):
        if name not in self.__callbacks:
            self.__callbacks[name] = [cb]
        else:
            self.__callbacks[name].append(cb)

    @property
    def training_dataset(self):
        return self.__training_dataset

    def set_training_dataset(self, training_dataset: torch.utils.data.Dataset):
        self.__training_dataset = training_dataset

    @property
    def validation_dataset(self):
        return self.__validation_dataset

    def set_validation_dataset(self, validation_dataset: torch.utils.data.Dataset):
        self.__validation_dataset = validation_dataset

    @property
    def test_dataset(self):
        return self.__test_dataset

    def set_test_dataset(self, test_dataset: torch.utils.data.Dataset):
        self.__test_dataset = test_dataset

    @property
    def device(self):
        return self.__device

    def set_device(self, device):
        self.__device = device

    def get_inferencer(
        self, phase: MachineLearningPhase, copy_model=True
    ) -> Inferencer:
        assert phase != MachineLearningPhase.Training

        dataset = self.validation_dataset
        if phase == MachineLearningPhase.Test:
            dataset = self.test_dataset
        if self.model_with_loss.model_type == ModelType.Classification:
            return ClassificationInferencer(
                self.model_with_loss,
                dataset,
                phase=phase,
                hyper_parameter=self.hyper_parameter,
                copy_model=copy_model,
                device=self.device,
            )
        if self.model_with_loss.model_type == ModelType.Detection:
            return DetectionInferencer(
                self.model_with_loss,
                dataset,
                phase=phase,
                hyper_parameter=self.hyper_parameter,
                iou_threshold=0.6,
                copy_model=copy_model,
                device=self.device,
            )
        assert False
        return None

    def set_hyper_parameter(self, hyper_parameter):
        self.__hyper_parameter = hyper_parameter

    @property
    def hyper_parameter(self):
        return self.__hyper_parameter

    def get_optimizer(self):
        if "optimizer" not in self.__data:
            self.set_data(
                "optimizer",
                self.hyper_parameter.get_optimizer(
                    self.model.parameters(), self.get_data("training_set_size")
                ),
            )
        return self.get_data("optimizer")

    def get_lr_scheduler(self):
        if "lr_scheduler" not in self.__data:
            self.set_data(
                "lr_scheduler",
                self.hyper_parameter.get_lr_scheduler(
                    self.get_optimizer(), self.get_data("training_set_size")
                ),
            )
        return self.get_data("lr_scheduler")

    def set_model(self, model: torch.nn.Module):
        self.model_with_loss.set_model(model)

    def load_model(self, model_path):
        self.set_model(torch.load(model_path, map_location=self.device))

    def save_model(self, save_dir, model_name="model.pt"):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model, os.path.join(save_dir, model_name))

    def repeated_train(self, repeated_num, save_dir=None, **kwargs):
        def training_callback(_, trainer: BasicTrainer):
            nonlocal save_dir, kwargs
            get_logger().setLevel(logging.ERROR)
            kwargs["test_epoch_interval"] = 1
            trainer.train(**kwargs)
            if save_dir is not None:
                trainer.save_model(save_dir)
            get_logger().setLevel(logging.DEBUG)
            return {
                "training_loss": trainer.training_loss,
                "validation_loss": trainer.validation_loss,
                "validation_accuracy": trainer.validation_accuracy,
                "test_loss": trainer.test_loss,
                "test_accuracy": trainer.test_accuracy,
            }

        return BasicTrainer.__repeated_training(repeated_num, self, training_callback)

    def train(self, **kwargs):
        training_set_size = len(self.training_dataset)
        self.set_data("training_set_size", training_set_size)
        get_logger().info("training_set_size is %s", training_set_size)
        get_logger().info("use device %s", self.device)
        self.__clear_loss()

        for callback in self.get_callbacks("pre_training_callbacks"):
            callback(self)
        for epoch in range(1, self.hyper_parameter.epoch + 1):
            optimizer = self.get_optimizer()
            lr_scheduler = self.get_lr_scheduler()
            assert optimizer is not None
            assert lr_scheduler is not None
            training_loss = 0.0
            for batch_index, batch in enumerate(
                self.__hyper_parameter.get_dataloader(
                    self.training_dataset, phase=MachineLearningPhase.Training
                )
            ):
                self.model_with_loss.set_model_mode(MachineLearningPhase.Training)
                self.model.to(self.device)
                optimizer.zero_grad()
                self.set_data(
                    "learning_rates", [group["lr"] for group in optimizer.param_groups]
                )
                for callback in self.get_callbacks("pre_batch_callbacks"):
                    callback(self, batch_index, batch)

                instance_inputs, instance_targets, _ = self.decode_batch(batch)
                optimizer.zero_grad()
                result = self.model_with_loss(
                    instance_inputs,
                    instance_targets,
                    phase=MachineLearningPhase.Training,
                )
                loss = result["loss"]
                loss.backward()
                batch_loss = loss.data.item()

                normalized_batch_loss = batch_loss
                if self.model_with_loss.is_averaged_loss():
                    real_batch_size = get_batch_size(instance_inputs)
                    normalized_batch_loss *= real_batch_size
                normalized_batch_loss /= training_set_size
                training_loss += normalized_batch_loss

                callbacks = kwargs.get("optimizer_step_callbacks", [])
                if callbacks:
                    for callback in callbacks:
                        callback(optimizer, trainer=self, device=self.device)
                else:
                    optimizer.step()
                if HyperParameter.lr_scheduler_step_after_batch(lr_scheduler):
                    get_logger().debug("adjust lr after batch")
                    lr_scheduler.step()

                for callback in self.get_callbacks("after_batch_callbacks"):
                    callback(
                        self,
                        batch_index,
                        batch=batch,
                        epoch=epoch,
                        batch_loss=batch_loss,
                    )

            self.training_loss.append(training_loss)
            for callback in self.get_callbacks("after_epoch_callbacks"):
                callback(
                    self,
                    epoch,
                    optimizer=optimizer,
                    **kwargs,
                )

            if not HyperParameter.lr_scheduler_step_after_batch(lr_scheduler):
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    get_logger().debug(
                        "call ReduceLROnPlateau for training loss %s",
                        self.training_loss[-1],
                    )
                    lr_scheduler.step(self.training_loss[-1])
                else:
                    lr_scheduler.step()

    # TODO:drop it and merge to dataset code
    def decode_batch(self, batch):
        instance_inputs = put_data_to_device(batch[0], self.device)
        instance_targets = put_data_to_device(batch[1], self.device)
        instance_indices = None
        if len(batch) >= 3:
            instance_indices = [idx.data.item() for idx in batch[2]]
        return (instance_inputs, instance_targets, instance_indices)

    def __clear_loss(self):
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
