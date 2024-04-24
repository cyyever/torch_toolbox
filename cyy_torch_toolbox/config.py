import copy
import datetime
import os
import uuid
from typing import Any

from cyy_naive_lib.log import log_debug, log_error, set_level
from omegaconf import OmegaConf

from .dataset import DatasetCollection, DatasetCollectionConfig
from .hyper_parameter import HyperParameterConfig
from .inferencer import Inferencer
from .ml_type import MachineLearningPhase
from .model import ModelConfig, ModelEvaluator
from .reproducible_env import ReproducibleEnvConfig
from .trainer import Trainer, TrainerConfig


class Config:
    def __init__(self, dataset_name: str = "", model_name: str = "") -> None:
        self.save_dir: str = ""
        self.log_level: Any = None
        self.reproducible_env_config = ReproducibleEnvConfig()
        self.dc_config: DatasetCollectionConfig = DatasetCollectionConfig(dataset_name)
        self.model_config = ModelConfig(model_name=model_name)
        self.hyper_parameter_config: HyperParameterConfig = HyperParameterConfig()
        self.trainer_config = TrainerConfig()

    def load_config(self, conf: Any, check_config: bool = True) -> dict:
        return self.__load_config(self, conf, check_config)

    def create_dataset_collection(self) -> DatasetCollection:
        log_debug("use dataset %s", self.dc_config.dataset_name)
        dc = self.dc_config.create_dataset_collection(
            save_dir=self.get_save_dir(),
        )
        return dc

    def create_trainer(
        self,
        dc: DatasetCollection | None = None,
        model_evaluator: ModelEvaluator | None = None,
    ) -> Trainer:
        if dc is None:
            dc = self.create_dataset_collection()
        if model_evaluator is None:
            model_evaluator = self.__create_model(dc)
        hyper_parameter = self.hyper_parameter_config.create_hyper_parameter()
        trainer: Trainer = self.trainer_config.create_trainer(
            dataset_collection=dc,
            model_evaluator=model_evaluator,
            hyper_parameter=hyper_parameter,
        )
        trainer.set_save_dir(self.get_save_dir())
        return trainer

    def create_inferencer(
        self,
        phase: MachineLearningPhase = MachineLearningPhase.Test,
        inherent_device: bool = True,
    ) -> Inferencer:
        trainer = self.create_trainer()
        return trainer.get_inferencer(phase=phase, inherent_device=inherent_device)

    def apply_global_config(self) -> None:
        if self.log_level is not None:
            set_level(self.log_level)
        self.reproducible_env_config.set_reproducible_env(self.get_save_dir())

    def __create_model(self, dc: DatasetCollection) -> ModelEvaluator:
        return self.model_config.get_model(dc)

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
        for attr in dir(obj):
            if "config" in attr:
                conf_container = cls.__load_config(
                    getattr(obj, attr), conf_container, check_config=False
                )
        if check_config:
            if conf_container:
                log_error("remain config %s", conf_container)
            assert not conf_container
        return conf_container

    def get_save_dir(self) -> str:
        if not self.save_dir:
            model_name = self.model_config.model_name
            if not model_name:
                model_name = "custom_model"
            date = datetime.datetime.now()
            self.save_dir = os.path.join(
                "session",
                self.dc_config.dataset_name,
                model_name,
                f"{date:%Y-%m-%d_%H_%M_%S}",
                str(uuid.uuid4()),
            )
        return self.save_dir
