import copy
import functools
import os
import sys

import torch
import torch.nn

from cyy_torch_toolbox.dependency import has_hugging_face
from cyy_torch_toolbox.ml_type import (DatasetType, MachineLearningPhase,
                                       ModelType)

if has_hugging_face:
    from cyy_torch_toolbox.models.huggingface_models import (
        get_hugging_face_model_constructors,
    )

from cyy_naive_lib.log import get_logger

from cyy_torch_toolbox.dataset_collection import DatasetCollection
from cyy_torch_toolbox.model_with_loss import (ModelEvaluator,
                                               get_model_with_loss)

__model_info: dict = {}


def _get_model_info() -> dict:
    if __model_info:
        return __model_info
    github_repos: dict[DatasetType, list] = {}
    github_repos[DatasetType.Vision] = [
        "pytorch/vision:main",
        "huggingface/pytorch-image-models:main",
    ]
    for dataset_type in DatasetType:
        if dataset_type not in github_repos:
            github_repos[dataset_type] = []
        github_repos[dataset_type].append("cyyever/torch_models:main")

    for dataset_type, repos in github_repos.items():
        for repo in repos:
            entrypoints = torch.hub.list(
                repo, force_reload=False, trust_repo=True, skip_validation=True
            )
            for model_name in entrypoints:
                if dataset_type not in __model_info:
                    __model_info[dataset_type] = {}
                if model_name.lower() not in __model_info[dataset_type]:
                    __model_info[dataset_type][model_name.lower()] = (
                        model_name,
                        functools.partial(
                            torch.hub.load,
                            repo_or_dir=repo,
                            model=model_name,
                            force_reload=False,
                            trust_repo=True,
                            skip_validation=True,
                            verbose=False,
                        ),
                        repo,
                    )
                else:
                    get_logger().debug("ignore model_name %s", model_name)

    if has_hugging_face:
        __model_info[DatasetType.Text] |= get_hugging_face_model_constructors()
    return __model_info


def get_model(
    name: str, dataset_collection: DatasetCollection, model_kwargs: dict = {}
) -> torch.nn.Module:
    model_info = _get_model_info()

    final_model_kwargs: dict = {}
    match dataset_collection.dataset_type:
        case DatasetType.Vision:
            dataset_util = dataset_collection.get_dataset_util()
            for k in ("input_channels", "channels"):
                if k not in model_kwargs:
                    final_model_kwargs |= {
                        k: dataset_util.channel,
                    }
        case DatasetType.Text:
            for k in ("num_embeddings", "token_num"):
                if k not in model_kwargs:
                    if dataset_collection.tokenizer is not None:
                        final_model_kwargs[k] = len(dataset_collection.tokenizer.vocab)
        case DatasetType.Graph:
            if "num_features" not in model_kwargs:
                final_model_kwargs[
                    "num_features"
                ] = dataset_collection.get_original_dataset(
                    phase=MachineLearningPhase.Training
                ).num_features

    model_type = ModelType.Classification
    if "rcnn" in name.lower():
        model_type = ModelType.Detection
    if "num_classes" not in model_kwargs:
        final_model_kwargs["num_classes"] = len(
            dataset_collection.get_labels(use_cache=True)
        )
        if model_type == ModelType.Detection:
            final_model_kwargs["num_classes"] += 1
    final_model_kwargs["num_labels"] = final_model_kwargs["num_classes"]
    final_model_kwargs |= model_kwargs
    # use_checkpointing = model_kwargs.pop("use_checkpointing", False)
    while True:
        try:
            model_constructors = model_info.get(dataset_collection.dataset_type, {})
            model_name, model_constructor, repo = model_constructors.get(
                name.lower(), (None, None, None)
            )
            if model_constructor is None:
                raise NotImplementedError(
                    f"unsupported model {name}, supported models are "
                    + str(model_constructors.keys())
                )
            get_logger().info("use model %s", model_name)
            model = model_constructor(**final_model_kwargs)
            get_logger().warning("use model arguments %s", final_model_kwargs)
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
            return model
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
    def __init__(self, model_name: str):
        self.model_name: str = model_name
        self.model_path = None
        self.pretrained: bool = False
        self.model_kwargs: dict = {}

    def get_model(self, dc: DatasetCollection) -> ModelEvaluator:
        assert not (self.pretrained and self.model_path)
        model_kwargs = copy.deepcopy(self.model_kwargs)
        if "pretrained" not in model_kwargs:
            model_kwargs["pretrained"] = self.pretrained
        if self.model_path is not None:
            assert "model_path" not in model_kwargs
            model_kwargs["model_path"] = self.model_path
        model = get_model(
            name=self.model_name, dataset_collection=dc, model_kwargs=model_kwargs
        )
        return get_model_with_loss(
            model=model, dataset_collection=dc, model_kwargs=model_kwargs
        )
