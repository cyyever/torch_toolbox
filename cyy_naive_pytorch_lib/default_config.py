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


class DefaultConfig:
    def __init__(self, dataset_name=None, model_name=None):
        self.make_reproducible = False
        self.reproducible_env_load_path = None
        self.dataset_name = dataset_name
        self.training_dataset_percentage = None
        self.training_dataset_indices_path = None
        self.training_dataset_label_map_path = None
        self.training_dataset_label_noise_percentage = None
        self.model_name = model_name
        self.epoch = None
        self.batch_size = None
        self.find_learning_rate = True
        self.learning_rate = None
        self.learning_rate_scheduler = None
        self.momentum = None
        self.weight_decay = None
        self.optimizer_name = None
        self.model_path = None
        self.save_dir = None
        self.log_level = None

    def load_args(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument("--dataset_name", type=str, required=True)
        parser.add_argument("--model_name", type=str, required=True)
        parser.add_argument("--epoch", type=int, default=None)
        parser.add_argument("--batch_size", type=int, default=None)
        parser.add_argument("--learning_rate", type=float, default=None)
        parser.add_argument("--learning_rate_scheduler", type=str, default=None)
        parser.add_argument("--find_learning_rate", action="store_true", default=False)
        parser.add_argument("--momentum", type=float, default=None)
        parser.add_argument("--weight_decay", type=float, default=None)
        parser.add_argument("--optimizer_name", type=str, default=None)
        parser.add_argument("--model_path", type=str, default=None)
        parser.add_argument("--save_dir", type=str, default=None)
        parser.add_argument("--reproducible_env_load_path", type=str, default=None)
        parser.add_argument("--make_reproducible", action="store_true", default=False)
        parser.add_argument("--training_dataset_percentage", type=float, default=None)
        parser.add_argument("--training_dataset_indices_path", type=str, default=None)
        parser.add_argument(
            "--training_dataset_label_noise_percentage", type=float, default=None
        )
        parser.add_argument("--log_level", type=str, default=None)
        parser.add_argument("--config_file", type=str, default=None)
        args = parser.parse_args()
        for attr in dir(args):
            if attr.startswith("_"):
                continue
            if hasattr(self, attr):
                value = getattr(args, attr)
                if value is not None:
                    setattr(self, attr, getattr(args, attr))
        return args

    def get_save_dir(self):
        if self.save_dir is None:
            self.save_dir = os.path.join(
                "session", self.dataset_name, self.model_name, str(uuid.uuid4())
            )
        os.makedirs(self.save_dir, exist_ok=True)
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
        if self.optimizer_name is not None:
            hyper_parameter.set_optimizer_factory(
                HyperParameter.get_optimizer_factory(self.optimizer_name)
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
        subset_indices = None
        if self.training_dataset_percentage is not None:
            subset_dict = DatasetUtil(training_dataset).iid_sample(
                self.training_dataset_percentage
            )
            subset_indices = sum(subset_dict.values(), [])
            with open(
                os.path.join(self.get_save_dir(), "training_dataset_indices.json"),
                mode="wt",
            ) as f:
                json.dump(subset_indices, f)

        if self.training_dataset_indices_path is not None:
            assert subset_indices is None
            get_logger().info(
                "use training_dataset_indices_path %s",
                self.training_dataset_indices_path,
            )
            with open(self.training_dataset_indices_path, "r") as f:
                subset_indices = json.load(f)
        if subset_indices is not None:
            training_dataset = sub_dataset(training_dataset, subset_indices)

        randomized_label_map = None
        if self.training_dataset_label_noise_percentage:
            randomized_label_map = DatasetUtil(training_dataset).randomize_subset_label(
                self.training_dataset_label_noise_percentage
            )
            with open(
                os.path.join(
                    self.get_save_dir(),
                    "randomized_label_map.json",
                ),
                mode="wt",
            ) as f:
                json.dump(randomized_label_map, f)

        if self.training_dataset_label_map_path is not None:
            assert randomized_label_map is not None
            get_logger().info(
                "use training_dataset_label_map_path %s",
                self.training_dataset_label_map_path,
            )
            randomized_label_map = json.load(
                open(self.training_dataset_label_map_path, "r")
            )

        if randomized_label_map is not None:
            training_dataset = replace_dataset_labels(
                training_dataset, randomized_label_map
            )
        return training_dataset

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
            global_reproducible_env.save(self.get_save_dir())
