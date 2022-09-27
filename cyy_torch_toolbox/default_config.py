import copy
import datetime
import os
import uuid
from dataclasses import dataclass

import torch
from cyy_naive_lib.log import get_logger
from omegaconf import OmegaConf

from cyy_torch_toolbox.dataset_collection import (DatasetCollection,
                                                  DatasetCollectionConfig)
from cyy_torch_toolbox.hyper_parameter import HyperParameterConfig
from cyy_torch_toolbox.inferencer import Inferencer
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.model_factory import ModelConfig
from cyy_torch_toolbox.model_with_loss import ModelWithLoss
from cyy_torch_toolbox.reproducible_env import global_reproducible_env
from cyy_torch_toolbox.trainer import Trainer


@dataclass
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
        self.cache_transforms = None
        self.use_amp = True
        self.benchmark_cudnn = True

    def load_config(self, conf, check_config: bool = True) -> dict:
        return DefaultConfig.__load_config(self, conf, check_config)

    @classmethod
    def __load_config(cls, obj, conf, check_config: bool = True) -> dict:
        if not isinstance(conf, dict):
            conf_container = OmegaConf.to_container(conf)
        else:
            conf_container = conf
        for attr in copy.copy(conf_container):
            if not hasattr(obj, attr):
                continue
            value = conf_container.pop(attr)
            if value is not None:
                match value:
                    case dict():
                        setattr(obj, attr, value | getattr(obj, attr))
                    case _:
                        setattr(obj, attr, value)
        if hasattr(obj, "dc_config"):
            conf_container = cls.__load_config(
                obj.dc_config, conf_container, check_config=False
            )
        if hasattr(obj, "hyper_parameter_config"):
            conf_container = cls.__load_config(
                obj.hyper_parameter_config, conf_container, check_config=False
            )
        if hasattr(obj, "model_config"):
            conf_container = cls.__load_config(
                obj.model_config, conf_container, check_config=False
            )
        if check_config:
            if conf_container:
                get_logger().error("remain config %s", conf_container)
            assert not conf_container
        return conf_container

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
        get_logger().debug("use dataset %s", self.dc_config.dataset_name)
        return self.dc_config.create_dataset_collection(
            save_dir=self.get_save_dir(),
            model_config=self.model_config,
        )

    def create_trainer(self, dc: DatasetCollection = None) -> Trainer:
        if dc is None:
            dc = self.create_dataset_collection()
        model_with_loss = self.model_config.get_model(dc)
        return self.create_trainer_by_model(model_with_loss, dc)

    def create_trainer_by_model(
        self, model_with_loss: ModelWithLoss, dc: DatasetCollection | None = None
    ) -> Trainer:
        if dc is None:
            dc = self.create_dataset_collection()
        if hasattr(dc, "adapt_to_model"):
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
        trainer.cache_transforms = self.cache_transforms
        if self.use_amp:
            trainer.set_amp()
        return trainer

    def create_inferencer(self, phase=MachineLearningPhase.Test) -> Inferencer:
        trainer = self.create_trainer()
        return trainer.get_inferencer(phase)

    def apply_global_config(self):
        if self.log_level is not None:
            get_logger().setLevel(self.log_level)
        if self.benchmark_cudnn:
            get_logger().info("benchmark cudnn")
        torch.backends.cudnn.benchmark = self.benchmark_cudnn
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
