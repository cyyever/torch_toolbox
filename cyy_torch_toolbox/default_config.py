import copy
import datetime
import os
import uuid
from dataclasses import dataclass
from typing import Any

import torch
from cyy_naive_lib.log import get_logger, set_level
from omegaconf import OmegaConf

from cyy_torch_toolbox.dataset_collection import (DatasetCollection,
                                                  DatasetCollectionConfig)
from cyy_torch_toolbox.hook_config import HookConfig
from cyy_torch_toolbox.hyper_parameter import HyperParameterConfig
from cyy_torch_toolbox.inferencer import Inferencer
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.model_evaluator import (ModelEvaluator,
                                               get_model_evaluator)
from cyy_torch_toolbox.model_factory import ModelConfig
from cyy_torch_toolbox.reproducible_env import global_reproducible_env
from cyy_torch_toolbox.trainer import Trainer


@dataclass
class DefaultConfig:
    def __init__(self, dataset_name: str, model_name: str) -> None:
        self.make_reproducible_env = False
        self.reproducible_env_load_path = None
        self.dc_config: DatasetCollectionConfig = DatasetCollectionConfig(dataset_name)
        self.hyper_parameter_config: HyperParameterConfig = HyperParameterConfig()
        self.model_config = ModelConfig(model_name=model_name)
        self.hook_config = HookConfig()
        self.save_dir: str | None = None
        self.log_level = None
        self.cache_transforms: None | str = None

    def load_config(self, conf: Any, check_config: bool = True) -> dict:
        return DefaultConfig.__load_config(self, conf, check_config)

    @classmethod
    def __load_config(cls, obj: Any, conf: Any, check_config: bool = True) -> dict:
        if not isinstance(conf, dict):
            conf_container: dict = OmegaConf.to_container(conf)
        else:
            conf_container = conf
        for attr in copy.copy(conf_container):
            assert isinstance(attr, str)
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
        if hasattr(obj, "hook_config"):
            conf_container = cls.__load_config(
                obj.hook_config, conf_container, check_config=False
            )
        if check_config:
            if conf_container:
                get_logger().error("remain config %s", conf_container)
            assert not conf_container
        return conf_container

    def get_save_dir(self) -> str:
        model_name = self.model_config.model_name
        if model_name is None:
            model_name = "custom_model"
        if self.save_dir is None:
            self.save_dir = os.path.join(
                "session",
                self.dc_config.dataset_name,
                model_name,
                "{date:%Y-%m-%d_%H_%M_%S}".format(date=datetime.datetime.now()),
                str(uuid.uuid4()),
            )
        return self.save_dir

    def create_dataset_collection(self) -> DatasetCollection:
        get_logger().debug("use dataset %s", self.dc_config.dataset_name)
        return self.dc_config.create_dataset_collection(
            save_dir=self.get_save_dir(),
        )

    def create_trainer(
        self,
        dc: DatasetCollection | None = None,
        model_evaluator: ModelEvaluator | None = None,
        model: None | torch.nn.Module = None,
    ) -> Trainer:
        assert not (model and model_evaluator)
        if dc is None:
            dc = self.create_dataset_collection()
        if model is not None:
            model_evaluator = get_model_evaluator(model, dc)

        if model_evaluator is None:
            model_evaluator = self.model_config.get_model(dc)
        dc.add_transforms(
            model_evaluator=model_evaluator,
            dataset_kwargs=dc.dataset_kwargs,
        )
        hyper_parameter = self.hyper_parameter_config.create_hyper_parameter(
            self.dc_config.dataset_name, self.model_config.model_name
        )
        trainer = Trainer(
            model_evaluator=model_evaluator,
            dataset_collection=dc,
            hyper_parameter=hyper_parameter,
            hook_config=self.hook_config,
        )
        trainer.set_save_dir(self.get_save_dir())
        trainer.cache_transforms = self.cache_transforms
        return trainer

    def create_inferencer(
        self, phase: MachineLearningPhase = MachineLearningPhase.Test
    ) -> Inferencer:
        trainer = self.create_trainer()
        return trainer.get_inferencer(phase)

    def apply_global_config(self) -> None:
        if self.log_level is not None:
            set_level(self.log_level)
        self.__set_reproducible_env()

    def __set_reproducible_env(self) -> None:
        if self.reproducible_env_load_path is not None:
            assert not global_reproducible_env.enabled
            global_reproducible_env.load(self.reproducible_env_load_path)
            self.make_reproducible_env = True

        if self.make_reproducible_env:
            global_reproducible_env.enable()
            if self.reproducible_env_load_path is None:
                global_reproducible_env.save(self.get_save_dir())
