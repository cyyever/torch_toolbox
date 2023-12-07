import copy
import os
import sys
from typing import Callable

import torch
from cyy_naive_lib.log import get_logger

from ..dataset.collection import DatasetCollection
from ..factory import Factory
from ..ml_type import DatasetType, ModelType
from .evaluator import ModelEvaluator, VisionModelEvaluator
from .repositary import get_model_info
from .util import ModelUtil  # noqa: F401

global_model_evaluator_factory = Factory()


def get_model_evaluator(
    model: torch.nn.Module, dataset_collection: DatasetCollection, **model_kwargs
) -> ModelEvaluator:
    model_evaluator_fun: Callable = ModelEvaluator
    if dataset_collection.dataset_type == DatasetType.Vision:
        model_evaluator_fun = VisionModelEvaluator
    else:
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


def get_model(
    name: str, dataset_collection: DatasetCollection, model_kwargs: dict
) -> dict:
    model_constructor_info = None
    constructor = None
    if dataset_collection.dataset_type in global_model_factory:
        constructor = global_model_factory[dataset_collection.dataset_type].get(
            name.lower()
        )
    if constructor is None:
        model_constructors = get_model_info().get(dataset_collection.dataset_type, {})
        model_constructor_info = model_constructors.get(name.lower(), {})
        if not model_constructor_info:
            raise NotImplementedError(
                f"unsupported model {name}, supported models are "
                + str(model_constructors.keys())
            )
        constructor = model_constructor_info["constructor"]

    final_model_kwargs: dict = {}
    match dataset_collection.dataset_type:
        case DatasetType.Vision:
            dataset_util = dataset_collection.get_dataset_util()
            for k in ("input_channels", "channels"):
                if k not in model_kwargs:
                    final_model_kwargs |= {
                        k: dataset_util.channel,
                    }

    final_model_kwargs |= model_kwargs
    model_type = ModelType.Classification
    if "rcnn" in name.lower():
        model_type = ModelType.Detection
    if model_type in (ModelType.Classification, ModelType.Detection):
        if "num_classes" not in final_model_kwargs:
            final_model_kwargs["num_classes"] = dataset_collection.label_number  # E:
            get_logger().debug("detect %s classes", final_model_kwargs["num_classes"])
        else:
            assert (
                final_model_kwargs["num_classes"] == dataset_collection.label_number
            )  # E:
    if model_type == ModelType.Detection:
        final_model_kwargs["num_classes"] += 1
    final_model_kwargs["num_labels"] = final_model_kwargs["num_classes"]
    final_model_kwargs["dataset_collection"] = dataset_collection
    # use_checkpointing = model_kwargs.pop("use_checkpointing", False)
    while True:
        try:
            model = constructor(**final_model_kwargs)
            if model_constructor_info is None:
                repo = None
            else:
                repo = model_constructor_info.get("repo", None)
            if repo is not None:
                # we need the model path to pickle models
                hub_dir = torch.hub.get_dir()
                repo_owner, repo_name, ref = torch.hub._parse_repo_info(repo)
                normalized_br = ref.replace("/", "_")
                owner_name_branch = "_".join([repo_owner, repo_name, normalized_br])
                repo_dir = os.path.join(hub_dir, owner_name_branch)
                sys.path.append(repo_dir)
            get_logger().debug(
                "use model arguments %s for model %s", final_model_kwargs, name
            )
            res = model
            if not isinstance(model, dict):
                res = {"model": model}
            return res
        except TypeError as e:
            retry = False
            for k in copy.copy(final_model_kwargs):
                if k in str(e):
                    get_logger().debug("%s so remove %s", e, k)
                    final_model_kwargs.pop(k)
                    retry = True
                    break
            if not retry:
                raise e


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
