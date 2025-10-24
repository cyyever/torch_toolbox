import functools

import torch
from cyy_naive_lib.log import log_debug
from cyy_naive_lib.reflection import get_kwarg_names

from ..ml_type import DatasetType


def get_torch_hub_model_info(repo: str) -> dict:
    model_info: dict = {}
    kwargs = {
        "force_reload": False,
        "trust_repo": True,
        "skip_validation": True,
    }

    if "verbose" in get_kwarg_names(torch.hub.list):
        kwargs["verbose"] = False

    torch_hub_models = torch.hub.list(repo, **kwargs)
    for model_name in torch_hub_models:
        if model_name.lower() not in model_info:
            model_info[model_name.lower()] = {
                "name": model_name,
                "constructor": functools.partial(
                    torch.hub.load,
                    repo_or_dir=repo,
                    model=model_name,
                    force_reload=False,
                    trust_repo=True,
                    skip_validation=True,
                    verbose=False,
                ),
                "repo": repo,
            }
        else:
            log_debug("ignore model_name %s", model_name)

    return model_info


def get_model_info() -> dict:
    model_info: dict = {}
    for dataset_type in DatasetType:
        model_info[dataset_type] = get_torch_hub_model_info(
            repo="cyyever/torch_models:main"
        )
    return model_info
