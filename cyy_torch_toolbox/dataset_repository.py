import functools

import torch
from cyy_naive_lib.storage import persistent_cache

from cyy_torch_toolbox.dependency import (has_hugging_face, has_medmnist,
                                          has_torch_geometric, has_torchaudio,
                                          has_torchtext, has_torchvision)

if has_torchvision:
    import torchvision

    import cyy_torch_toolbox.dataset_wrapper.vision as local_vision_datasets

if has_torchtext:
    import torchtext

if has_torch_geometric:
    import torch_geometric
if has_torchaudio:
    import torchaudio

    import cyy_torch_toolbox.dataset_wrapper.audio as local_audio_datasets
if has_medmnist:
    import medmnist
if has_hugging_face:
    import datasets

from cyy_naive_lib.reflection import get_class_attrs

from cyy_torch_toolbox.ml_type import DatasetType


@persistent_cache(cache_time=3600 * 24)
def get_dataset_constructors(cache_path: str) -> dict:
    repositories = {}
    if has_torchvision:
        repositories[DatasetType.Vision] = [
            torchvision.datasets,
            local_vision_datasets,
        ]
    if has_torchtext:
        repositories[DatasetType.Text] = [torchtext.datasets]
    if has_torchaudio:
        repositories[DatasetType.Audio] = [torchaudio.datasets, local_audio_datasets]
    if has_torch_geometric:
        repositories[DatasetType.Graph] = [torch_geometric.datasets]
    dataset_constructors: dict = {}
    for dataset_type, repositorys in repositories.items():
        dataset_constructors[dataset_type] = {}
        for repository in repositorys:
            if hasattr(repository, "DATASETS"):
                for name, dataset_constructor in getattr(
                    repository, "DATASETS"
                ).items():
                    if dataset_type == DatasetType.Text:
                        dataset_constructors[dataset_type][name] = dataset_constructor
                continue
            dataset_constructors[dataset_type] |= get_class_attrs(
                repository,
                filter_fun=lambda k, v: issubclass(v, torch.utils.data.Dataset),
            )
    if has_torch_geometric:
        graph_dataset_constructors = dataset_constructors[DatasetType.Graph]
        if "Planetoid" in graph_dataset_constructors:
            for repository in ["Cora", "CiteSeer", "PubMed"]:
                graph_dataset_constructors[repository] = functools.partial(
                    graph_dataset_constructors["Planetoid"],
                    name=repository,
                )

    if has_medmnist:
        INFO = medmnist.info.INFO
        for name, item in INFO.items():
            medmnist_cls = getattr(medmnist, item["python_class"])
            setattr(medmnist_cls, "targets", item["label"])
            dataset_constructors[DatasetType.Vision][name] = functools.partial(
                medmnist_cls, target_transform=lambda x: x[0]
            )
    if has_hugging_face:
        dataset_names = set(
            a.lower() for a in dataset_constructors[DatasetType.Text].keys()
        )
        for name in datasets.list_datasets(
            with_community_datasets=False, with_details=False
        ):
            if name.lower() not in dataset_names:
                dataset_constructors[DatasetType.Text][name] = functools.partial(
                    datasets.load_dataset, name
                )
    return dataset_constructors
