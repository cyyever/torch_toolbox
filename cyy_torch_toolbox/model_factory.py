import copy
import functools
import os
import sys

import torch
import torch.nn

try:
    import transformers

    has_hugging_face = True
except BaseException:
    has_hugging_face = False
from cyy_naive_lib.log import get_logger

from cyy_torch_toolbox.dataset_collection import DatasetCollection
from cyy_torch_toolbox.ml_type import DatasetType, ModelType
from cyy_torch_toolbox.model_with_loss import (ModelWithLoss,
                                               TextModelWithLoss,
                                               VisionModelWithLoss)
from cyy_torch_toolbox.models.huggingface_models import huggingface_models

__model_info: dict = {}


def get_model_info() -> dict:
    if __model_info:
        return __model_info
    github_repos = [
        "pytorch/vision:main",
        "cyyever/torch_models:main",
        "huggingface/pytorch-image-models:main",
    ]

    for repo in github_repos:
        try:
            entrypoints = torch.hub.list(
                repo, force_reload=False, trust_repo=True, skip_validation=True
            )
        except BaseException:
            entrypoints = torch.hub.list(repo, force_reload=False, skip_validation=True)
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

    if has_hugging_face:
        for model_name in huggingface_models:
            full_model_name = "sequence_classification_" + model_name

            def create_model(model_name, pretrained, **model_kwargs):
                pretrained_model = (
                    transformers.AutoModelForSequenceClassification.from_pretrained(
                        model_name, **model_kwargs
                    )
                )
                if pretrained:
                    return pretrained_model
                get_logger().warning("use huggingface without pretrained parameters")
                old_embedding = pretrained_model.get_input_embeddings()
                config = transformers.AutoConfig.from_pretrained(
                    model_name, **model_kwargs
                )
                model = transformers.AutoModelForSequenceClassification.from_config(
                    config
                )
                model.set_input_embeddings(old_embedding)
                return model

            if full_model_name.lower() not in __model_info:
                __model_info[full_model_name.lower()] = (
                    full_model_name,
                    functools.partial(
                        create_model,
                        model_name,
                    ),
                    None,
                )
    return __model_info


def get_model(
    name: str, dataset_collection: DatasetCollection, model_kwargs: dict = {}
) -> torch.nn.Module:
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
    added_kwargs["num_classes"] = model_kwargs.get("num_classes", None)
    if added_kwargs["num_classes"] is None:
        added_kwargs["num_classes"] = len(dataset_collection.get_labels(use_cache=True))
        if model_type == ModelType.Detection:
            added_kwargs["num_classes"] += 1
    added_kwargs["num_labels"] = added_kwargs["num_classes"]
    # use_checkpointing = model_kwargs.pop("use_checkpointing", False)
    while True:
        try:
            model_name, model_constructor, repo = model_info.get(
                name.lower(), (None, None, None)
            )
            if model_constructor is None:
                raise NotImplementedError(
                    f"unsupported model {name}, supported models are "
                    + str(model_info.keys())
                )
            get_logger().info("use model %s", model_name)
            model = model_constructor(**(added_kwargs | model_kwargs))
            get_logger().warning("use model arguments %s", model_kwargs | added_kwargs)
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


def get_model_with_loss(
    model: torch.nn.Module,
    dataset_collection: DatasetCollection,
    model_type: None | ModelType = None,
    model_kwargs: dict = {},
) -> ModelWithLoss:
    model_with_loss_fun = ModelWithLoss
    if dataset_collection.dataset_type == DatasetType.Vision:
        model_with_loss_fun = VisionModelWithLoss
    elif dataset_collection.dataset_type == DatasetType.Text:
        model_with_loss_fun = TextModelWithLoss
    loss_fun_name = model_kwargs.get("loss_fun_name", None)
    model_type = ModelType.Classification
    model_with_loss = model_with_loss_fun(
        model=model,
        loss_fun=loss_fun_name,
        model_type=model_type,
    )
    model_path = model_kwargs.get("model_path", None)
    if model_path is not None:
        model_with_loss.model.load_state_dict(torch.load(model_path))
    word_vector_name = model_kwargs.get("word_vector_name", None)
    if word_vector_name is not None:
        from cyy_torch_toolbox.word_vector import PretrainedWordVector

        PretrainedWordVector(word_vector_name).load_to_model(
            model_with_loss=model_with_loss, tokenizer=dataset_collection.tokenizer
        )
    return model_with_loss


class ModelConfig:
    def __init__(self, model_name: str):
        self.model_name: str = model_name
        self.model_path = None
        self.pretrained: bool = False
        self.model_kwargs: dict = {}

    def get_model(self, dc: DatasetCollection) -> ModelWithLoss:
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
