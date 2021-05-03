import argparse
import datetime
import os
import uuid

from cyy_naive_lib.log import get_logger

from dataset_collection import DatasetCollectionConfig
from hyper_parameter import HyperParameterConfig
from inference import Inferencer
from ml_type import MachineLearningPhase
from model_factory import get_model
from reproducible_env import global_reproducible_env
from trainer import Trainer


class DefaultConfig:
    def __init__(self, dataset_name=None, model_name=None):
        self.make_reproducible = False
        self.reproducible_env_load_path = None
        self.dc_config: DatasetCollectionConfig = DatasetCollectionConfig(dataset_name)
        self.hyper_parameter_config: HyperParameterConfig = HyperParameterConfig()
        self.model_name = model_name
        self.model_path = None
        self.pretrained = False
        self.debug = False
        self.save_dir = None
        self.log_level = None

    def load_args(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()

        parser.add_argument("--model_name", type=str, required=True)
        parser.add_argument("--model_path", type=str, default=None)
        parser.add_argument("--pretrained", action="store_true", default=False)
        parser.add_argument("--save_dir", type=str, default=None)
        parser.add_argument("--reproducible_env_load_path", type=str, default=None)
        parser.add_argument("--make_reproducible", action="store_true", default=False)
        self.dc_config.add_args(parser)
        self.hyper_parameter_config.add_args(parser)
        parser.add_argument("--log_level", type=str, default=None)
        parser.add_argument("--debug", action="store_true", default=False)
        args = parser.parse_args()
        self.dc_config.load_args(args)
        self.hyper_parameter_config.load_args(args)

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
        hyper_parameter = self.hyper_parameter_config.create_hyper_parameter(
            self.dc_config.dataset_name, self.model_name
        )

        dc = self.dc_config.create_dataset_collection(self.get_save_dir())
        model_with_loss = get_model(self.model_name, dc, pretrained=self.pretrained)
        trainer = Trainer(
            model_with_loss, dc, hyper_parameter, save_dir=self.get_save_dir()
        )
        if self.debug:
            trainer.debug_mode = True
        if self.model_path is not None:
            trainer.load_model(self.model_path)

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
