import copy
import os
import sys

import torch
import torch.nn
from cyy_naive_lib.log import get_logger

from ..dataset_collection import DatasetCollection
from ..factory import Factory
from ..ml_type import DatasetType, MachineLearningPhase, ModelType
from ..model_evaluator import ModelEvaluator, get_model_evaluator
from .torch_model import get_torch_model_info

globel_model_factory = Factory()


def get_model(
    name: str, dataset_collection: DatasetCollection, model_kwargs: dict
) -> dict:
    constructor = globel_model_factory.get(dataset_collection.dataset_type)
    if constructor is not None:
        return constructor(
            name=name, dataset_collection=dataset_collection, model_kwargs=model_kwargs
        )
    model_constructors = get_torch_model_info().get(dataset_collection.dataset_type, {})
    model_constructor_info = model_constructors.get(name.lower(), {})
    if not model_constructor_info:
        raise NotImplementedError(
            f"unsupported model {name}, supported models are "
            + str(model_constructors.keys())
        )

    final_model_kwargs: dict = {}
    match dataset_collection.dataset_type:
        case DatasetType.Vision:
            dataset_util = dataset_collection.get_dataset_util()
            for k in ("input_channels", "channels"):
                if k not in model_kwargs:
                    final_model_kwargs |= {
                        k: dataset_util.channel,
                    }
        case DatasetType.Graph:
            if "num_features" not in model_kwargs:
                final_model_kwargs[
                    "num_features"
                ] = dataset_collection.get_original_dataset(
                    phase=MachineLearningPhase.Training
                ).num_features

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
    # use_checkpointing = model_kwargs.pop("use_checkpointing", False)
    while True:
        try:
            model = model_constructor_info["constructor"](**final_model_kwargs)
            get_logger().debug(
                "use model arguments %s for model %s",
                final_model_kwargs,
                model_constructor_info["name"],
            )
            res = {"model": model}
            repo = model_constructor_info.get("repo", None)
            if repo is not None:
                # we need the model path to pickle models
                hub_dir = torch.hub.get_dir()
                # Parse github repo information
                repo_owner, repo_name, ref = torch.hub._parse_repo_info(repo)
                # Github allows branch name with slash '/',
                # this causes confusion with path on both Linux and Windows.
                # Backslash is not allowed in Github branch name so no need to
                # to worry about it.
                normalized_br = ref.replace("/", "_")
                # Github renames folder repo-v1.x.x to repo-1.x.x
                # We don't know the repo name before downloading the zip file
                # and inspect name from it.
                # To check if cached repo exists, we need to normalize folder names.
                owner_name_branch = "_".join([repo_owner, repo_name, normalized_br])
                repo_dir = os.path.join(hub_dir, owner_name_branch)
                sys.path.append(repo_dir)
            return res
        except TypeError as e:
            retry = False
            for k in copy.copy(final_model_kwargs):
                if k in str(e):
                    get_logger().debug("%s so remove %s", e, k)
                    final_model_kwargs.pop(k)
                    retry = True
                    break
            # if not retry:
            #     if "pretrained" in str(e) and not model_kwargs["pretrained"]:
            #         model_kwargs.pop("pretrained")
            #         retry = True
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
