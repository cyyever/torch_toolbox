import copy
import os
import sys
from typing import Callable

import torch
from cyy_naive_lib.log import log_debug

from ..dataset import ClassificationDatasetCollection, DatasetCollection
from ..factory import Factory
from ..ml_type import DatasetType, ModelType
from .amp import AMPModelEvaluator
from .evaluator import ModelEvaluator
from .util import ModelUtil

__all__ = ["AMPModelEvaluator", "ModelUtil"]

global_model_evaluator_factory = Factory()


def get_model_evaluator(
    model: torch.nn.Module,
    dataset_collection: DatasetCollection | None = None,
    **model_kwargs,
) -> ModelEvaluator:
    model_evaluator_fun = ModelEvaluator
    if dataset_collection is not None:
        model_evaluator_fun = global_model_evaluator_factory.get(
            dataset_collection.dataset_type
        )
    model_evaluator = model_evaluator_fun(
        model=model,
        loss_fun=model_kwargs.pop("loss_fun_name", None),
        dataset_collection=dataset_collection,
        **model_kwargs,
    )
    return model_evaluator


global_model_factory: dict[DatasetType, Factory] = {}


def create_model(constructor, **kwargs):
    while True:
        try:
            res = constructor(**kwargs)
            log_debug("use model arguments %s for model", kwargs)
            return res
        except TypeError as e:
            retry = False
            for k in copy.copy(kwargs):
                if k in str(e):
                    log_debug("%s so remove %s", e, k)
                    kwargs.pop(k, None)
                    retry = True
                    break
            if not retry:
                raise e


def get_model(
    name: str, dataset_collection: DatasetCollection, model_kwargs: dict
) -> dict:
    model_kwargs = copy.copy(model_kwargs)
    model_constructor: Callable | None = global_model_factory[
        dataset_collection.dataset_type
    ].get(name.lower())
    if model_constructor is None:
        raise NotImplementedError(f"unsupported model {name}")

    model_type = ModelType.Classification
    if "rcnn" in name.lower():
        model_type = ModelType.Detection
    if model_type in (ModelType.Classification, ModelType.Detection):
        assert isinstance(dataset_collection, ClassificationDatasetCollection)
        if "num_classes" not in model_kwargs:
            model_kwargs["num_classes"] = dataset_collection.label_number  # E:
            log_debug("detect %s classes", model_kwargs["num_classes"])
        else:
            assert model_kwargs["num_classes"] == dataset_collection.label_number  # E:
    if model_type == ModelType.Detection:
        model_kwargs["num_classes"] += 1
    model_kwargs["num_labels"] = model_kwargs["num_classes"]
    model_kwargs["dataset_collection"] = dataset_collection

    res = model_constructor(**model_kwargs)
    repo = res.get("repo", None)
    if repo is not None:
        # we need the model path to pickle models
        hub_dir = torch.hub.get_dir()
        repo_owner, repo_name, ref = torch.hub._parse_repo_info(repo)
        normalized_br = ref.replace("/", "_")
        owner_name_branch = "_".join([repo_owner, repo_name, normalized_br])
        repo_dir = os.path.join(hub_dir, owner_name_branch)
        sys.path.append(repo_dir)
    if not isinstance(res, dict):
        res = {"model": res}
    return res


class ModelConfig:
    def __init__(self, model_name: str) -> None:
        self.model_name: str = model_name
        self.model_kwargs: dict = {}

    def get_model(self, dc: DatasetCollection) -> ModelEvaluator:
        self.model_kwargs["name"] = self.model_name
        model_kwargs = copy.deepcopy(self.model_kwargs)
        if "pretrained" not in model_kwargs:
            model_kwargs["pretrained"] = False
        model_res = get_model(
            name=self.model_name,
            dataset_collection=dc,
            model_kwargs=model_kwargs,
        )
        model_evaluator = get_model_evaluator(
            dataset_collection=dc,
            model_name=self.model_name,
            **(model_kwargs | model_res),
        )
        return model_evaluator
