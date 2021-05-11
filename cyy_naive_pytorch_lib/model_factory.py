import os
import sys

import torch
import torchvision
from cyy_naive_lib.log import get_logger

from dataset_collection import DatasetCollection
from ml_type import ModelType
from model_with_loss import ModelWithLoss


def list_local_models(local_dir):
    sys.path.insert(0, local_dir)

    hub_module = torch.hub.import_module(
        torch.hub.MODULE_HUBCONF, local_dir + "/" + torch.hub.MODULE_HUBCONF
    )

    sys.path.remove(local_dir)

    # We take functions starts with '_' as internal helper functions
    entrypoints = [
        f
        for f in dir(hub_module)
        if callable(getattr(hub_module, f)) and not f.startswith("_")
    ]

    return entrypoints


__model_info: dict = dict()


def get_model_info():
    global __model_info
    repos = [
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "models"),
    ]
    github_repos = ["pytorch/vision", "lukemelas/EfficientNet-PyTorch"]

    if not __model_info:
        for repo in repos:
            for model_name in list_local_models(repo):
                if model_name not in __model_info:
                    __model_info[model_name.lower()] = (repo, model_name, "local")
        for repo in github_repos:
            for model_name in torch.hub.list(repo):
                if model_name not in __model_info:
                    __model_info[model_name.lower()] = (repo, model_name, "github")
    return __model_info


def get_model(
    name: str, dataset_collection: DatasetCollection, **model_kwargs
) -> ModelWithLoss:

    model_info = get_model_info()
    model_repo_and_name = model_info.get(name.lower(), None)
    if model_repo_and_name is None:
        get_logger().error(
            "Unknown model name: %s. These models are supported:%s",
            name,
            list(model_info.keys()),
        )
        raise RuntimeError("unknown model name:", name)

    dataset_util = dataset_collection.get_dataset_util()
    added_kwargs = {
        "input_channels": dataset_util.channel,
        "channels": dataset_util.channel,
        "num_classes": len(dataset_collection.get_labels()),
    }
    model_type = ModelType.Classification
    # FIXME: use more robust method to determine detection models
    if "rcnn" in name.lower():
        model_type = ModelType.Detection
    if model_type == ModelType.Detection:
        added_kwargs["num_classes"] += 1
    repo, model_name, source = model_repo_and_name
    loss_fun_name = model_kwargs.pop("loss_fun_name", None)
    for k in list(added_kwargs.keys()):
        if k in model_kwargs:
            added_kwargs.pop(k)

    while True:
        try:
            model_with_loss = ModelWithLoss(
                model=torch.hub.load(
                    repo, model_name, source=source, **(added_kwargs | model_kwargs)
                ),
                loss_fun=loss_fun_name,
                model_type=model_type,
            )
            input_size = getattr(model_with_loss.model.__class__, "input_size", None)
            if input_size is not None:
                get_logger().warning("use input_size %s", input_size)
                model_with_loss.append_transform(
                    torchvision.transforms.Resize(input_size)
                )
            get_logger().warning("use model arguments %s", model_kwargs | added_kwargs)
            return model_with_loss
        except TypeError as e:
            retry = False
            for k in added_kwargs:
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
