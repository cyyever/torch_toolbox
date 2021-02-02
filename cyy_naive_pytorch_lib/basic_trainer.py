# import copy
import logging
import os
# from enum import Enum, auto
from typing import Callable, Dict, List, Union

import torch
from cyy_naive_lib.log import get_logger

from dataset_collection import DatasetCollection
from hyper_parameter import HyperParameter
from inference import ClassificationInferencer, DetectionInferencer, Inferencer
from ml_types import MachineLearningPhase, ModelType
from model_executor import ModelExecutor, ModelExecutorCallbackPoint
from model_loss import ModelWithLoss
from tensor import get_batch_size


class StopTrainingException(Exception):
    pass


class BasicTrainer(ModelExecutor):
    def __init__(
        self,
        model_with_loss: ModelWithLoss,
        dataset_collection: DatasetCollection,
        hyper_parameter: HyperParameter,
    ):
        super().__init__(model_with_loss, dataset_collection, hyper_parameter)
        self.__clear_loss()
        self.add_callback(
            ModelExecutorCallbackPoint.BEFORE_BATCH,
            lambda trainer, batch, batch_index: trainer.set_data(
                "cur_learning_rates",
                [group["lr"] for group in trainer.get_optimizer().param_groups],
            ),
        )

    @property
    def training_dataset(self) -> torch.utils.data.Dataset:
        return self.dataset_collection.get_dataset(MachineLearningPhase.Training)

    def get_inferencer(
        self, phase: MachineLearningPhase, copy_model=True
    ) -> Inferencer:
        assert phase != MachineLearningPhase.Training

        if self.model_with_loss.model_type == ModelType.Classification:
            return ClassificationInferencer(
                self.model_with_loss,
                self.dataset_collection,
                phase=phase,
                hyper_parameter=self.hyper_parameter,
                copy_model=copy_model,
            )
        if self.model_with_loss.model_type == ModelType.Detection:
            return DetectionInferencer(
                self.model_with_loss,
                self.dataset_collection,
                phase=phase,
                hyper_parameter=self.hyper_parameter,
                iou_threshold=0.6,
                copy_model=copy_model,
            )
        assert False
        return None

    def get_optimizer(self):
        if not self.has_data("optimizer"):
            self.set_data(
                "optimizer",
                self.hyper_parameter.get_optimizer(self),
            )
        return self.get_data("optimizer")

    def get_lr_scheduler(self):
        if not self.has_data("lr_scheduler"):
            self.set_data(
                "lr_scheduler",
                self.hyper_parameter.get_lr_scheduler(self),
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
        self.exec_callbacks(ModelExecutorCallbackPoint.BEFORE_TRAINING, self)
        try:
            for epoch in range(1, self.hyper_parameter.epoch + 1):
                optimizer = self.get_optimizer()
                lr_scheduler = self.get_lr_scheduler()
                assert optimizer is not None
                assert lr_scheduler is not None
                training_loss = 0.0
                for batch_index, batch in enumerate(
                    self.dataset_collection.get_dataloader(
                        phase=MachineLearningPhase.Training,
                        hyper_parameter=self.hyper_parameter,
                    )
                ):
                    self.model_with_loss.set_model_mode(MachineLearningPhase.Training)
                    self.model.to(self.device)
                    optimizer.zero_grad()
                    self.exec_callbacks(
                        ModelExecutorCallbackPoint.BEFORE_BATCH,
                        self,
                        batch_index,
                        batch,
                    )
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

                    if self.has_callback(ModelExecutorCallbackPoint.OPTIMIZER_STEP):
                        self.exec_callbacks(
                            ModelExecutorCallbackPoint.OPTIMIZER_STEP, optimizer, self
                        )
                    else:
                        optimizer.step()

                    if HyperParameter.lr_scheduler_step_after_batch(lr_scheduler):
                        get_logger().debug("adjust lr after batch")
                        lr_scheduler.step()

                    self.exec_callbacks(
                        ModelExecutorCallbackPoint.AFTER_BATCH,
                        self,
                        batch_index,
                        batch=batch,
                        epoch=epoch,
                        batch_loss=batch_loss,
                    )

                self.training_loss.append(training_loss)
                self.exec_callbacks(
                    ModelExecutorCallbackPoint.AFTER_EPOCH,
                    self,
                    epoch,
                    optimizer=optimizer,
                )

                if not HyperParameter.lr_scheduler_step_after_batch(lr_scheduler):
                    if isinstance(
                        lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        get_logger().debug(
                            "call ReduceLROnPlateau for training loss %s",
                            self.training_loss[-1],
                        )
                        lr_scheduler.step(self.training_loss[-1])
                    else:
                        lr_scheduler.step()
        except StopTrainingException as e:
            get_logger().warning("stop training:%s", e)

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
