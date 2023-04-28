import copy
import os
import sys

import torch
import torch.nn
from cyy_naive_lib.log import get_logger

from ..dataset_collection import DatasetCollection
from ..dependency import has_hugging_face
from ..ml_type import DatasetType, MachineLearningPhase, ModelType
from ..model_evaluator import ModelEvaluator, get_model_evaluator
from .torch_model import get_torch_model_info

if has_hugging_face:
    from .huggingface_model import get_hugging_face_model_info


def __get_model_info() -> dict:
    model_info = get_torch_model_info()
    for dataset_type, info in get_hugging_face_model_info().items():
        if dataset_type not in model_info:
            model_info[dataset_type] = info
        else:
            for model_name, res in info.items():
                if model_name in model_info[dataset_type]:
                    raise RuntimeError(f"model {model_name} from multiple sources")
                model_info[dataset_type][model_name] = res
    return model_info


def get_model(
    name: str,
    dataset_collection: DatasetCollection,
    model_kwargs: dict | None = None,
) -> torch.nn.Module:
    model_constructors = __get_model_info().get(dataset_collection.dataset_type, {})
    model_constructor_info = model_constructors.get(name.lower(), {})
    if not model_constructor_info:
        raise NotImplementedError(
            f"unsupported model {name}, supported models are "
            + str(model_constructors.keys())
        )

    if not model_kwargs:
        model_kwargs = {}
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
            model = model_constructor_info["constructor"](**final_model_kwargs)
            get_logger().warning(
                "use model arguments %s for model %s",
                final_model_kwargs,
                model_constructor_info["name"],
            )
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

    def get_model(
        self, dc: DatasetCollection, dataset_kwargs: None | dict = None
    ) -> ModelEvaluator:
        assert not (self.pretrained and self.model_path)
        model_kwargs = copy.deepcopy(self.model_kwargs)
        if "pretrained" not in model_kwargs:
            model_kwargs["pretrained"] = self.pretrained
        if self.model_path is not None:
            assert "model_path" not in model_kwargs
            model_kwargs["model_path"] = self.model_path
        model = get_model(
            name=self.model_name,
            dataset_collection=dc,
            model_kwargs=model_kwargs,
        )
        return get_model_evaluator(
            model=model,
            dataset_collection=dc,
            model_name=self.model_name,
            model_kwargs=model_kwargs,
        )
