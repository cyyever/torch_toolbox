import copy

import torch
from cyy_naive_lib.log import get_logger

from dataset_collection import DatasetCollection
from ml_type import DatasetType, ModelType
from model_with_loss import ModelWithLoss

__model_info: dict = {}


def get_model_info() -> dict:
    global __model_info
    github_repos = [
        "pytorch/vision:main",
        "cyyever/torch_models:main",
        # "huggingface/transformers:master",
        "lukemelas/EfficientNet-PyTorch:master",
    ]

    if not __model_info:
        for repo in github_repos:
            for model_name in torch.hub.list(repo):
                if model_name not in __model_info:
                    __model_info[model_name.lower()] = (repo, model_name, "github")
    return __model_info


def get_model(
    name: str, dataset_collection: DatasetCollection, **model_kwargs
) -> ModelWithLoss:
    model_info = get_model_info()

    dataset_util = dataset_collection.get_dataset_util()
    if model_kwargs is None or "num_classes" not in model_kwargs:
        added_kwargs = {
            "num_classes": len(dataset_collection.get_labels(use_cache=True)),
        }
    if dataset_collection.dataset_type == DatasetType.Vision:
        added_kwargs |= {
            "input_channels": dataset_util.channel,
            "channels": dataset_util.channel,
        }
    # if dataset_collection.dataset_type == DatasetType.Text:
    #     added_kwargs["num_embeddings"] = len(
    #         dataset_collection.tokenizer_and_vocab.vocab
    #     )

    model_type = ModelType.Classification
    if "rcnn" in name.lower():
        model_type = ModelType.Detection
    if model_type == ModelType.Detection:
        added_kwargs["num_classes"] += 1
    loss_fun_name = model_kwargs.pop("loss_fun_name", None)

    while True:
        try:
            model_repo_and_name = model_info.get(name.lower(), None)
            if model_repo_and_name is None:
                raise NotImplementedError(
                    f"unsupported model {name}, supported models are "
                    + str(model_info.keys())
                )
            repo, model_name, source = model_repo_and_name
            model = torch.hub.load(
                repo, model_name, source=source, **(added_kwargs | model_kwargs)
            )
            model_with_loss = ModelWithLoss(
                model=model,
                loss_fun=loss_fun_name,
                model_type=model_type,
            )
            get_logger().warning("use model arguments %s", model_kwargs | added_kwargs)
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
