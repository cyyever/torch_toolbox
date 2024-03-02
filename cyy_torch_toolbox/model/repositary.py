import functools

import torch
import torch.nn
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.reflection import get_kwarg_names

from ..ml_type import DatasetType


def get_model_info() -> dict:
    model_info: dict = {}
    github_repos: dict[DatasetType, list] = {}
    for dataset_type in DatasetType:
        if dataset_type not in github_repos:
            github_repos[dataset_type] = []
        github_repos[dataset_type].append("cyyever/torch_models:main")

    for dataset_type, repos in github_repos.items():
        for repo in repos:
            kwargs = {
                "force_reload": False,
                "trust_repo": True,
                "skip_validation": True,
            }

            if "verbose" in get_kwarg_names(torch.hub.list):
                kwargs["verbose"] = False

            torch_hub_models = torch.hub.list(repo, **kwargs)
            for model_name in torch_hub_models:
                if dataset_type not in model_info:
                    model_info[dataset_type] = {}
                if model_name.lower() not in model_info[dataset_type]:
                    model_info[dataset_type][model_name.lower()] = {
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
                    get_logger().debug("ignore model_name %s", model_name)

    return model_info
