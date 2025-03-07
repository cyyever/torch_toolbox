import copy
import datetime
import os
import sys
import uuid
from typing import Any

import hydra
from cyy_naive_lib.log import log_debug, log_error, set_level
from omegaconf import OmegaConf

from .dataset import DatasetCollection, DatasetCollectionConfig
from .hyper_parameter import HyperParameterConfig
from .inferencer import Inferencer
from .ml_type import ConfigBase, MachineLearningPhase
from .model import ModelConfig
from .reproducible_env import ReproducibleEnvConfig
from .trainer import Trainer, TrainerConfig


class Config(ConfigBase):
    def __init__(self, dataset_name: str = "", model_name: str = "") -> None:
        super().__init__()
        self.save_dir: str = ""
        self.log_level: Any | None = None
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
    ) -> Trainer:
        hyper_parameter = self.hyper_parameter_config.create_hyper_parameter()
        trainer: Trainer = self.trainer_config.create_trainer(
            dataset_collection_config=self.dc_config,
            model_config=self.model_config,
            hyper_parameter=hyper_parameter,
        )
        if dc is not None:
            trainer.set_dataset_collection(dc)
        trainer.set_save_dir(self.get_save_dir())
        return trainer

    def create_inferencer(
        self,
        dc: DatasetCollection | None = None,
        phase: MachineLearningPhase = MachineLearningPhase.Test,
    ) -> Inferencer:
        inferencer = self.create_trainer().get_inferencer(phase=phase, copy_model=False)
        if dc is not None:
            inferencer.set_dataset_collection(dc)
        return inferencer

    def apply_global_config(self) -> None:
        if self.log_level is not None:
            set_level(self.log_level)
        self.reproducible_env_config.set_reproducible_env(self.get_save_dir())

    @classmethod
    def __load_config(cls, obj: Any, conf: Any, check_config: bool = True) -> dict:
        if not isinstance(conf, dict):
            conf_container: Any = OmegaConf.to_container(conf)
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

    def fix_paths(self, project_path: str) -> None:
        for k, v in self.dc_config.dataset_kwargs.items():
            if k not in ("train_files", "test_files", "validation_files"):
                continue
            data_dir = self.dc_config.dataset_kwargs.get(
                "data_dir", os.path.join(project_path, "data")
            )
            if k == "train_files":
                data_dir = self.dc_config.dataset_kwargs.get("train_data_dir", data_dir)
            elif k == "test_files":
                data_dir = self.dc_config.dataset_kwargs.get("test_data_dir", data_dir)
            files = v
            if isinstance(v, str):
                files = [v]
            new_files = []
            for file in files:
                if not file.startswith("/"):
                    file = str(os.path.join(data_dir, file))
                    assert os.path.isfile(file), file
                new_files.append(file)
            self.dc_config.dataset_kwargs[k] = new_files


def load_config_from_hydra(
    config_path: str | None = None, other_config_files: list[str] | None = None
) -> Any:
    # disable hydra output dir
    for option in [
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
    ]:
        sys.argv.append(option)
    if other_config_files is None:
        other_config_files = []
    other_confs = [OmegaConf.load(file) for file in other_config_files]
    if config_path is not None:
        conf_obj: Any = None

        @hydra.main(config_path=config_path, version_base=None)
        def load_config_hydra(conf) -> None:
            nonlocal conf_obj
            conf_obj = conf

        load_config_hydra()

        other_confs.append(conf_obj)
    assert other_confs
    for idx, conf_obj in enumerate(other_confs):
        while "dataset_name" not in conf_obj and len(conf_obj) == 1:
            conf_obj = next(iter(conf_obj.values()))
        other_confs[idx] = conf_obj

    result_conf = other_confs[0]
    for o in other_confs[1:]:
        result_conf.merge_with(o)
    return result_conf
