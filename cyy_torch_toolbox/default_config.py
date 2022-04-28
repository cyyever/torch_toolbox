import argparse
import datetime
import os
import uuid

from cyy_naive_lib.log import get_logger

from cyy_torch_toolbox.dataset_collection import (DatasetCollection,
                                                  DatasetCollectionConfig)
from cyy_torch_toolbox.hyper_parameter import HyperParameterConfig
from cyy_torch_toolbox.inferencer import Inferencer
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.model_factory import ModelConfig
from cyy_torch_toolbox.model_with_loss import ModelWithLoss
from cyy_torch_toolbox.reproducible_env import global_reproducible_env
from cyy_torch_toolbox.trainer import Trainer


class DefaultConfig:
    def __init__(self, dataset_name=None, model_name=None):
        self.make_reproducible_env = False
        self.reproducible_env_load_path = None
        self.dc_config: DatasetCollectionConfig = DatasetCollectionConfig(dataset_name)
        self.hyper_parameter_config: HyperParameterConfig = HyperParameterConfig()
        self.model_config = ModelConfig(model_name=model_name)
        self.debug = False
        self.profile = False
        self.save_dir = None
        self.log_level = None

    def load_args(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()

        parser.add_argument("--save_dir", type=str, default=None)
        parser.add_argument("--reproducible_env_load_path", type=str, default=None)
        parser.add_argument(
            "--make_reproducible_env", action="store_true", default=False
        )
        self.dc_config.add_args(parser)
        self.hyper_parameter_config.add_args(parser)
        self.model_config.add_args(parser)
        parser.add_argument("--log_level", type=str, default=None)
        parser.add_argument("--debug", action="store_true", default=False)
        parser.add_argument("--profile", action="store_true", default=False)
        args = parser.parse_args()
        self.dc_config.load_args(args)
        self.hyper_parameter_config.load_args(args)
        self.model_config.load_args(args)

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
                self.model_config.model_name,
                "{date:%Y-%m-%d_%H_%M_%S}".format(date=datetime.datetime.now()),
                str(uuid.uuid4()),
            )
        os.makedirs(self.save_dir, exist_ok=True)
        return self.save_dir

    def create_dataset_collection(self):
        get_logger().info("use dataset %s", self.dc_config.dataset_name)
        return self.dc_config.create_dataset_collection(self.get_save_dir())

    def create_trainer(self, dc: DatasetCollection = None) -> Trainer:
        if dc is None:
            dc = self.create_dataset_collection()
        model_with_loss = self.model_config.get_model(dc)
        return self.create_trainer_by_model(model_with_loss, dc)

    def create_trainer_by_model(
        self, model_with_loss: ModelWithLoss, dc: DatasetCollection = None
    ) -> Trainer:
        if dc is None:
            dc = self.create_dataset_collection()
        dc.adapt_to_model(
            model_with_loss.get_real_model(), self.model_config.model_kwargs
        )
        hyper_parameter = self.hyper_parameter_config.create_hyper_parameter(
            self.dc_config.dataset_name, self.model_config.model_name
        )
        trainer = Trainer(model_with_loss, dc, hyper_parameter)
        trainer.set_save_dir(self.get_save_dir())
        if self.debug:
            get_logger().warning("debug the trainer")
            trainer.debugging_mode = True
        if self.profile:
            get_logger().warning("profile the trainer")
            trainer.profiling_mode = True
        return trainer

    def create_inferencer(self, phase=MachineLearningPhase.Test) -> Inferencer:
        trainer = self.create_trainer()
        return trainer.get_inferencer(phase)

    def apply_global_config(self):
        if self.log_level is not None:
            get_logger().setLevel(self.log_level)
        self.__set_reproducible_env()

    def __set_reproducible_env(self):
        if self.reproducible_env_load_path is not None:
            assert not global_reproducible_env.enabled
            global_reproducible_env.load(self.reproducible_env_load_path)
            self.make_reproducible_env = True

        if self.make_reproducible_env:
            global_reproducible_env.enable()
            if self.reproducible_env_load_path is None:
                global_reproducible_env.save(self.get_save_dir())
