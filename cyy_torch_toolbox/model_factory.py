import copy
import functools
import json
import os
import sys

import torch
import transformers
from cyy_naive_lib.log import get_logger

from cyy_torch_toolbox.dataset_collection import DatasetCollection
from cyy_torch_toolbox.ml_type import DatasetType, ModelType
from cyy_torch_toolbox.model_with_loss import (CheckPointedModelWithLoss,
                                               ModelWithLoss)
from cyy_torch_toolbox.models.huggingface_models import huggingface_models
from cyy_torch_toolbox.word_vector import PretrainedWordVector

__model_info: dict = {}


def get_model_info() -> dict:
    github_repos = [
        "pytorch/vision:main",
        "cyyever/torch_models:main",
        "rwightman/pytorch-image-models:master",
    ]

    if not __model_info:
        for repo in github_repos:
            try:
                entrypoints = torch.hub.list(
                    repo, force_reload=False, trust_repo=True, skip_validation=True
                )
            except BaseException:
                entrypoints = torch.hub.list(
                    repo, force_reload=False, skip_validation=True
                )
            for model_name in entrypoints:
                if model_name.lower() not in __model_info:
                    __model_info[model_name.lower()] = (
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
        for model_name in huggingface_models:
            full_model_name = "sequence_classification_" + model_name
            if full_model_name.lower() not in __model_info:
                __model_info[full_model_name.lower()] = (
                    full_model_name,
                    functools.partial(
                        transformers.AutoModelForSequenceClassification.from_pretrained,
                        model_name,
                    ),
                    None,
                )
    return __model_info


def get_model(
    name: str, dataset_collection: DatasetCollection, **model_kwargs
) -> ModelWithLoss:
    model_info = get_model_info()

    added_kwargs: dict = {}
    if dataset_collection.dataset_type == DatasetType.Vision:
        dataset_util = dataset_collection.get_dataset_util()
        for k in ("input_channels", "channels"):
            if k not in model_kwargs:
                added_kwargs |= {
                    k: dataset_util.channel,
                }
    if dataset_collection.dataset_type == DatasetType.Text:
        if "num_embeddings" not in model_kwargs:
            if dataset_collection.tokenizer is not None:
                added_kwargs["num_embeddings"] = len(dataset_collection.tokenizer.vocab)
        else:
            added_kwargs["num_embeddings"] = model_kwargs["num_embeddings"]
        if "token_num" not in model_kwargs:
            added_kwargs["token_num"] = added_kwargs["num_embeddings"]

    model_type = ModelType.Classification
    if "rcnn" in name.lower():
        model_type = ModelType.Detection
    try:
        added_kwargs["num_classes"] = model_kwargs.get("num_classes", None)
        if added_kwargs["num_classes"] is None:
            added_kwargs["num_classes"] = len(
                dataset_collection.get_labels(use_cache=True)
            )
            if model_type == ModelType.Detection:
                added_kwargs["num_classes"] += 1
        added_kwargs["num_labels"] = added_kwargs["num_classes"]
    except Exception:
        pass
    loss_fun_name = model_kwargs.pop("loss_fun_name", None)
    while True:
        try:
            _, model_constructor, repo = model_info.get(
                name.lower(), (None, None, None)
            )
            if model_constructor is None:
                raise NotImplementedError(
                    f"unsupported model {name}, supported models are "
                    + str(model_info.keys())
                )
            model = model_constructor(**(added_kwargs | model_kwargs))
            get_logger().warning("use model arguments %s", model_kwargs | added_kwargs)
            if model_kwargs.get("use_checkpointing", False):
                model_with_loss_fun = CheckPointedModelWithLoss
            else:
                model_with_loss_fun = ModelWithLoss
            model_with_loss = model_with_loss_fun(
                model=model,
                loss_fun=loss_fun_name,
                model_type=model_type,
            )
            if repo is not None:
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


class ModelConfig:
    def __init__(self, model_name=None):
        self.model_name = model_name
        self.model_path = None
        self.pretrained = False
        self.model_kwargs = {}
        self.model_kwarg_json_path = None

    def add_args(self, parser) -> None:
        if self.model_name is None:
            parser.add_argument("--model_name", type=str, required=True)
        parser.add_argument("--model_path", type=str, default=None)
        parser.add_argument("--pretrained", action="store_true", default=False)
        parser.add_argument("--model_kwarg_json_path", type=str, default=None)

    def load_args(self, args) -> None:
        for attr in dir(args):
            if attr.startswith("_"):
                continue
            if not hasattr(self, attr):
                continue
            get_logger().debug("set dataset collection config attr %s", attr)
            value = getattr(args, attr)
            if value is not None:
                setattr(self, attr, value)
        if args.model_kwarg_json_path is not None:
            with open(args.model_kwarg_json_path, "rt", encoding="utf-8") as f:
                self.model_kwargs |= json.load(f)

    def get_model(self, dc: DatasetCollection) -> ModelWithLoss:
        model_kwargs = copy.deepcopy(self.model_kwargs)
        if self.pretrained:
            model_kwargs["pretrained"] = True
        get_logger().info("use model %s", self.model_name)
        word_vector_name = model_kwargs.pop("word_vector_name", None)
        model_with_loss = get_model(self.model_name, dc, **model_kwargs)
        if self.model_path is not None:
            model_with_loss.model.load_state_dict(torch.load(self.model_path))
        if word_vector_name is not None:
            PretrainedWordVector(word_vector_name).load_to_model(
                model_with_loss=model_with_loss, tokenizer=dc.tokenizer
            )
        return model_with_loss
