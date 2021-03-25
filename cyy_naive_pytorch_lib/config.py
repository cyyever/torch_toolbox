import argparse
import copy
import json
import os
import uuid

import torch
from cyy_naive_lib.log import get_logger

from dataset import DatasetUtil, replace_dataset_labels, sub_dataset
from dataset_collection import DatasetCollection
from hyper_parameter import (HyperParameter, HyperParameterAction,
                             get_recommended_hyper_parameter)
from inference import Inferencer
from ml_type import MachineLearningPhase
from model_factory import get_model
from reproducible_env import global_reproducible_env
from trainer import Trainer


class Config:
    def __init__(self):
        self.make_reproducible = False
        self.reproducible_env_load_path = None
        self.dataset_name = None
        self.model_name = None
        self.epoch = None
        self.batch_size = None
        self.find_learning_rate = True
        self.learning_rate = None
        self.learning_rate_scheduler = None
        self.momentum = None
        self.weight_decay = None
        self.optimizer = None
        self.model_path = None
        self.save_dir = None
        self.training_dataset_percentage = None
        self.randomized_label_map_path = None
        self.training_dataset_indices_path = None
        self.log_level = None

    def get_save_dir(self):
        if self.save_dir is None:
            self.save_dir = os.path.join(
                "session", self.dataset_name, self.model_name, str(uuid.uuid4())
            )
        return self.save_dir

    def create_trainer(self, apply_env_factor=True) -> Trainer:
        if self.dataset_name is None:
            raise RuntimeError("dataset_name is None")
        if self.model_name is None:
            raise RuntimeError("model_name is None")
        get_logger().info(
            "use dataset %s and model %s", self.dataset_name, self.model_name
        )
        if apply_env_factor:
            self.__apply_env_config()
        # trainer = get_trainer_from_configuration(self.dataset_name, self.model_name)

        hyper_parameter = get_recommended_hyper_parameter(
            self.dataset_name, self.model_name
        )
        assert hyper_parameter is not None

        dc = DatasetCollection.get_by_name(self.dataset_name)
        model_with_loss = get_model(self.model_name, dc)
        trainer = Trainer(model_with_loss, dc, hyper_parameter)
        trainer.dataset_collection.transform_dataset(
            MachineLearningPhase.Training,
            self.__transform_training_dataset,
        )
        if self.model_path is not None:
            trainer.load_model(self.model_path)

        hyper_parameter = copy.deepcopy(trainer.hyper_parameter)
        assert hyper_parameter is not None
        if self.epoch is not None:
            hyper_parameter.set_epoch(self.epoch)
        if self.batch_size is not None:
            hyper_parameter.set_batch_size(self.batch_size)
        if self.learning_rate is not None and self.find_learning_rate:
            raise RuntimeError(
                "can't not specify a learning_rate and find a learning_rate at the same time"
            )
        if self.learning_rate is not None:
            hyper_parameter.set_learning_rate(self.learning_rate)
        if self.find_learning_rate:
            hyper_parameter.set_learning_rate(HyperParameterAction.FIND_LR)
        if self.momentum is not None:
            hyper_parameter.set_momentum(self.momentum)
        if self.weight_decay is not None:
            hyper_parameter.set_weight_decay(self.weight_decay)
        if self.optimizer is not None:
            hyper_parameter.set_optimizer_factory(
                HyperParameter.get_optimizer_factory(self.optimizer)
            )
        if self.learning_rate_scheduler is not None:
            hyper_parameter.set_lr_scheduler_factory(
                HyperParameter.get_lr_scheduler_factory(self.learning_rate_scheduler)
            )
        trainer.set_hyper_parameter(hyper_parameter)
        return trainer

    def create_inferencer(self, phase=MachineLearningPhase.Test) -> Inferencer:
        trainer = self.create_trainer()
        return trainer.get_inferencer(phase)

    def __transform_training_dataset(
        self, training_dataset
    ) -> torch.utils.data.Dataset:
        if self.training_dataset_percentage is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            subset_dict = DatasetUtil(training_dataset).sample_subset(
                self.training_dataset_percentage
            )
            sample_indices: list = sum(subset_dict.values(), [])
            training_dataset = sub_dataset(training_dataset, sample_indices)
            with open(
                os.path.join(self.save_dir, "training_dataset_indices.json"),
                mode="wt",
            ) as f:
                json.dump(sample_indices, f)

        if self.training_dataset_indices_path is not None:
            get_logger().info("use training_dataset_indices_path")
            with open(self.training_dataset_indices_path, "r") as f:
                subset_indices = json.load(f)
                training_dataset = sub_dataset(training_dataset, subset_indices)
        if self.randomized_label_map_path is not None:
            get_logger().info("use randomized_label_map_path")
            training_dataset = replace_dataset_labels(
                training_dataset, self.__get_randomized_label_map()
            )
        return training_dataset

    def __get_randomized_label_map(self):
        randomized_label_map: dict = dict()
        with open(self.randomized_label_map_path, "r") as f:
            for k, v in json.load(f).items():
                randomized_label_map[int(k)] = int(v)
        return randomized_label_map

    def __apply_env_config(self):
        if self.log_level is not None:
            get_logger().setLevel(self.log_level)
        self.__set_reproducible_env()

    def __set_reproducible_env(self):
        if self.reproducible_env_load_path is not None:
            assert not global_reproducible_env.initialized
            global_reproducible_env.load(self.reproducible_env_load_path)
            self.make_reproducible = True

        if self.make_reproducible:
            global_reproducible_env.enable()
            global_reproducible_env.save(self.save_dir)
