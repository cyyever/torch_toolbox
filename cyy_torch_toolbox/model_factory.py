import copy
import os
import sys

import torch
from cyy_naive_lib.log import get_logger

from dataset_collection import DatasetCollection
from ml_type import DatasetType, ModelType
from model_with_loss import ModelWithLoss

__model_info: dict = {}


def get_model_info() -> dict:
    github_repos = [
        "pytorch/vision:main",
        "cyyever/torch_models:main",
        # "huggingface/transformers:master",
        "lukemelas/EfficientNet-PyTorch:master",
    ]

    if not __model_info:
        for repo in github_repos:
            try:
                for model_name in torch.hub.list(
                    repo, force_reload=True, trust_repo=True, skip_validation=True
                ):
                    if model_name not in __model_info:
                        __model_info[model_name.lower()] = (
                            repo,
                            model_name,
                        )
            except BaseException as e:
                get_logger().warning("ignore exception %s", e)

    return __model_info


def get_model(
    name: str, dataset_collection: DatasetCollection, **model_kwargs
) -> ModelWithLoss:
    model_info = get_model_info()

    dataset_util = dataset_collection.get_dataset_util()
    added_kwargs: dict = {}
    if dataset_collection.dataset_type == DatasetType.Vision:
        for k in ("input_channels", "channels"):
            if k not in model_kwargs:
                added_kwargs |= {
                    k: dataset_util.channel,
                }
    if dataset_collection.dataset_type == DatasetType.Text:
        if "num_embeddings" not in model_kwargs:
            added_kwargs["num_embeddings"] = len(dataset_collection.tokenizer.vocab)

    model_type = ModelType.Classification
    if "rcnn" in name.lower():
        model_type = ModelType.Detection
    try:
        if "num_classes" not in model_kwargs:
            added_kwargs |= {
                "num_classes": len(dataset_collection.get_labels(use_cache=True)),
            }
            if model_type == ModelType.Detection:
                added_kwargs["num_classes"] += 1
    except Exception:
        pass
    loss_fun_name = model_kwargs.pop("loss_fun_name", None)
    while True:
        try:
            model_repo_and_name = model_info.get(name.lower(), None)
            if model_repo_and_name is None:
                raise NotImplementedError(
                    f"unsupported model {name}, supported models are "
                    + str(model_info.keys())
                )
            repo, model_name = model_repo_and_name
            model = torch.hub.load(
                repo,
                model_name,
                force_reload=False,
                trust_repo=True,
                skip_validation=True,
                **(added_kwargs | model_kwargs),
            )
            model_with_loss = ModelWithLoss(
                model=model,
                loss_fun=loss_fun_name,
                model_type=model_type,
            )
            get_logger().warning("use model arguments %s", model_kwargs | added_kwargs)
            # we need the path to pickle models
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
            return model_with_loss
        except TypeError as e:
            retry = False
            for k in copy.copy(added_kwargs):
                if k in str(e):
                    get_logger().debug("%s so remove %s", e, k)
                    added_kwargs.pop(k)
                    retry = True
                    break
            if not retry:
                if "pretrained" in str(e) and not model_kwargs["pretrained"]:
                    model_kwargs.pop("pretrained")
                    retry = True
            if not retry:
                raise e
