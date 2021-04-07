import argparse
import datetime
import os
import uuid

from cyy_naive_lib.log import get_logger

from dataset_collection import DatasetCollectionConfig
from hyper_parameter import (HyperParameter, HyperParameterAction,
                             get_recommended_hyper_parameter)
from inference import Inferencer
from ml_type import MachineLearningPhase
from model_factory import get_model
from reproducible_env import global_reproducible_env
from trainer import Trainer


class DefaultConfig:
    def __init__(self, dataset_name, model_name):
        self.make_reproducible = False
        self.reproducible_env_load_path = None
        self.dc_config = DatasetCollectionConfig(dataset_name)
        # self.dataset_name = dataset_name
        # self.dataset_args = dict()
        # self.training_dataset_percentage = None
        # self.training_dataset_indices_path = None
        # self.training_dataset_label_map_path = None
        # self.training_dataset_label_map = None
        # self.training_dataset_label_noise_percentage = None
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
        self.dc_config.add_args(parser)
        # parser.add_argument("--dataset_name", type=str, required=True)
        # parser.add_argument("--training_dataset_percentage", type=float, default=None)
        # parser.add_argument("--training_dataset_indices_path", type=str, default=None)
        # parser.add_argument(
        #     "--training_dataset_label_noise_percentage", type=float, default=None
        # )
        parser.add_argument("--log_level", type=str, default=None)
        parser.add_argument("--config_file", type=str, default=None)
        args = parser.parse_args()
        self.dc_config.load_args(args)

        for attr in dir(args):
            if attr.startswith("_"):
                continue
            value = getattr(args, attr)
            if value is not None:
                setattr(self, attr, value)
        return args

    def get_save_dir(self):
        if self.save_dir is None:
            self.save_dir = os.path.join(
                "session",
                self.dc_config.dataset_name,
                self.model_name,
                "{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now()),
                str(uuid.uuid4()),
            )
        os.makedirs(self.save_dir, exist_ok=True)
        return self.save_dir

    def create_trainer(self, apply_env_factor=True) -> Trainer:
        get_logger().info(
            "use dataset %s and model %s", self.dc_config.dataset_name, self.model_name
        )
        if apply_env_factor:
            self.__apply_env_config()
        hyper_parameter = get_recommended_hyper_parameter(
            self.dc_config.dataset_name, self.model_name
        )
        assert hyper_parameter is not None

        dc = self.dc_config.create_dataset_collection(self.get_save_dir())
        model_with_loss = get_model(self.model_name, dc)
        trainer = Trainer(
            model_with_loss, dc, hyper_parameter, save_dir=self.get_save_dir()
        )
        if self.model_path is not None:
            trainer.load_model(self.model_path)

        if self.epoch is not None:
            trainer.hyper_parameter.set_epoch(self.epoch)
        if self.batch_size is not None:
            trainer.hyper_parameter.set_batch_size(self.batch_size)
        if self.learning_rate is not None and self.find_learning_rate:
            raise RuntimeError(
                "can't not specify a learning_rate and find a learning_rate at the same time"
            )
        if self.learning_rate is not None:
            trainer.hyper_parameter.set_learning_rate(self.learning_rate)
        if self.find_learning_rate:
            trainer.hyper_parameter.set_learning_rate(HyperParameterAction.FIND_LR)
        if self.momentum is not None:
            trainer.hyper_parameter.set_momentum(self.momentum)
        if self.weight_decay is not None:
            trainer.hyper_parameter.set_weight_decay(self.weight_decay)
        if self.optimizer_name is not None:
            trainer.hyper_parameter.set_optimizer_factory(
                HyperParameter.get_optimizer_factory(self.optimizer_name)
            )
        if self.learning_rate_scheduler is not None:
            trainer.hyper_parameter.set_lr_scheduler_factory(
                HyperParameter.get_lr_scheduler_factory(self.learning_rate_scheduler)
            )
        return trainer

    def create_inferencer(self, phase=MachineLearningPhase.Test) -> Inferencer:
        trainer = self.create_trainer()
        return trainer.get_inferencer(phase)

    def __apply_env_config(self):
        if self.log_level is not None:
            get_logger().setLevel(self.log_level)
        self.__set_reproducible_env()

    def __set_reproducible_env(self):
        if self.reproducible_env_load_path is not None:
            if not global_reproducible_env.enabled:
                global_reproducible_env.load(self.reproducible_env_load_path)
            self.make_reproducible = True

        if self.make_reproducible:
            global_reproducible_env.enable()
            global_reproducible_env.save(self.get_save_dir())
