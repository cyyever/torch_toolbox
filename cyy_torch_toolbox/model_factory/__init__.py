import copy
import functools
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


@functools.cache
def __get_model_info() -> dict:
    model_info = get_torch_model_info()
    if has_hugging_face:
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
    name: str, dataset_collection: DatasetCollection, model_kwargs: dict
) -> torch.nn.Module:
    model_constructors = __get_model_info().get(dataset_collection.dataset_type, {})
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
        case DatasetType.Text:
            if dataset_collection.tokenizer is not None:
                for k in ("num_embeddings", "token_num"):
                    if k not in model_kwargs:
                        final_model_kwargs[k] = len(dataset_collection.tokenizer.vocab)
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
    def __init__(self, model_name: str) -> None:
        self.model_name: str = model_name
        self.model_kwargs: dict | None = None
        self.frozen_modules: dict | None = None

    def get_model(self, dc: DatasetCollection) -> ModelEvaluator:
        model_kwargs = copy.deepcopy(self.model_kwargs)
        if model_kwargs is None:
            model_kwargs = {}
        if "pretrained" not in model_kwargs:
            model_kwargs["pretrained"] = False
        if hasattr(dc, "set_model_kwargs"):
            dc.set_model_kwargs(model_kwargs | {"name": self.model_name})
        model = get_model(
            name=self.model_name,
            dataset_collection=dc,
            model_kwargs=model_kwargs,
        )
        model_evaluator = get_model_evaluator(
            model=model,
            dataset_collection=dc,
            model_name=self.model_name,
            model_kwargs=model_kwargs,
        )
        if self.frozen_modules is not None:
            match self.frozen_modules:
                case {"types": types}:
                    for t in types:
                        model_evaluator.model_util.freeze_modules(module_type=t)
                case {"names": names}:
                    for name in names:
                        model_evaluator.model_util.freeze_modules(module_name=name)
                case _:
                    raise NotImplementedError(self.frozen_modules)
        return model_evaluator
