import functools

import torch
import torchvision

try:
    import torchtext

    has_torchtext = True
except ModuleNotFoundError:
    has_torchtext = False

try:
    import torchaudio

    has_torchaudio = True
except ModuleNotFoundError:
    has_torchaudio = False

if has_torchaudio:
    import cyy_torch_toolbox.datasets.audio as local_audio_datasets

try:
    import medmnist

    has_medmnist = True
except ModuleNotFoundError:
    has_medmnist = False

try:
    import datasets

    has_hugging_face = True
except ModuleNotFoundError:
    has_hugging_face = False


from cyy_naive_lib.reflection import get_class_attrs

import cyy_torch_toolbox.datasets.vision as local_vision_datasets
from cyy_torch_toolbox.ml_type import DatasetType


def get_dataset_constructors(dataset_type: DatasetType = None) -> dict:
    repositories = []
    if dataset_type is None or dataset_type == DatasetType.Vision:
        repositories += [torchvision.datasets, local_vision_datasets]
    if dataset_type is None or dataset_type == DatasetType.Text:
        if has_torchtext:
            repositories += [torchtext.datasets]
    if dataset_type is None or dataset_type == DatasetType.Audio:
        if has_torchaudio:
            repositories += [torchaudio.datasets, local_audio_datasets]
    dataset_constructors = {}
    for repository in repositories:
        if hasattr(repository, "DATASETS"):
            for name, dataset_constructor in getattr(repository, "DATASETS").items():
                if dataset_type == DatasetType.Text:
                    dataset_constructors[name] = dataset_constructor
            continue
        dataset_constructors |= get_class_attrs(
            repository,
            filter_fun=lambda k, v: issubclass(v, torch.utils.data.Dataset),
        )
    if has_medmnist and (dataset_type is None or dataset_type == DatasetType.Vision):
        INFO = medmnist.info.INFO
        for name, item in INFO.items():
            medmnist_cls = getattr(medmnist, item["python_class"])
            setattr(medmnist_cls, "targets", item["label"])
            dataset_constructors[name] = functools.partial(
                medmnist_cls, target_transform=lambda x: x[0]
            )
    if has_hugging_face and (dataset_type is None or dataset_type == DatasetType.Text):
        dataset_names = set(a.lower() for a in dataset_constructors.keys())

        for name in datasets.list_datasets(
            with_community_datasets=False, with_details=False
        ):
            if name.lower() not in dataset_names:
                dataset_constructors[name] = functools.partial(
                    datasets.load_dataset, name
                )
    return dataset_constructors
