import copy
import os
import sys
from collections.abc import Callable

import torch
from cyy_naive_lib.log import log_debug

from ..dataset import ClassificationDatasetCollection, DatasetCollection
from ..ml_type import DatasetType, Factory, ModelType
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
    model_evaluator_funs: type | list[type] = ModelEvaluator
    if dataset_collection is not None:
        model_evaluator_funs = global_model_evaluator_factory.get(
            dataset_collection.dataset_type
        )
    if not isinstance(model_evaluator_funs, list):
        model_evaluator_funs = [model_evaluator_funs]
    for model_evaluator_fun in model_evaluator_funs:
        model_evaluator = model_evaluator_fun(
            model=model,
            loss_fun=model_kwargs.pop("loss_fun_name", None),
            dataset_collection=dataset_collection,
            **model_kwargs,
        )
        if model_evaluator is not None:
            return model_evaluator
    raise RuntimeError(f"No model evaluator for {model.name}")


global_model_factory: dict[DatasetType, list[Factory]] = {}


def create_model(constructor, **kwargs) -> Callable:
    while True:
        try:
            res = constructor(**kwargs)
            log_debug("use model arguments %s for model", kwargs)
            return res
        except TypeError as e:
            retry = False
            for k in copy.copy(kwargs):
                if f"got an unexpected keyword argument '{k}'" in str(e):
                    log_debug("%s so remove %s", e, k)
                    kwargs.pop(k, None)
                    retry = True
                    break
            if not retry:
                raise e


def get_model(
    name: str, model_kwargs: dict, dataset_collection: DatasetCollection | None = None
) -> dict:
    model_kwargs = copy.copy(model_kwargs)
    factories = []
    if dataset_collection is not None:
        factories = global_model_factory.get(dataset_collection.dataset_type, [])
    else:
        for v in global_model_factory.values():
            factories += v
    model_constructor: Callable | None | dict = None

    for factory in factories:
        model_constructor = factory.get(name)
        if model_constructor is not None:
            break
    if model_constructor is None:
        for factory in factories:
            model_constructor = factory.get(name.lower())
            if model_constructor is not None:
                break
    if model_constructor is None:
        raise NotImplementedError(f"unsupported model {name}")

    model_type: ModelType | None = None
    if isinstance(model_constructor, dict):
        model_type = model_constructor.get("model_type", model_type)
        model_constructor = model_constructor["constructor"]
    if model_type is None:
        if "rcnn" in name.lower():
            model_type = ModelType.Detection
        else:
            model_type = ModelType.Classification
    if model_type in (ModelType.Classification, ModelType.Detection):
        assert isinstance(dataset_collection, ClassificationDatasetCollection)
        if "num_classes" not in model_kwargs:
            model_kwargs["num_classes"] = dataset_collection.label_number
            log_debug("detect %s classes", model_kwargs["num_classes"])
        else:
            assert model_kwargs["num_classes"] == dataset_collection.label_number  # E:
        if model_type == ModelType.Detection:
            model_kwargs["num_classes"] += 1
        model_kwargs["num_labels"] = model_kwargs["num_classes"]
    if model_type in (ModelType.TokenClassification,):
        assert isinstance(dataset_collection, ClassificationDatasetCollection)
        model_kwargs["num_labels"] = dataset_collection.label_number
    if dataset_collection is not None:
        model_kwargs["dataset_collection"] = dataset_collection
    assert not isinstance(model_constructor, dict)
    assert model_constructor is not None
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
