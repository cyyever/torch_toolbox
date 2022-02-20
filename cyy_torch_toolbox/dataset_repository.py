import functools
import inspect

import torch

try:
    import torchaudio

    has_torchaudio = True
except ModuleNotFoundError:
    has_torchaudio = False
import torchtext
import torchvision

if has_torchaudio:
    import cyy_torch_toolbox.datasets.audio as local_audio_datasets

try:
    import medmnist

    has_medmnist = True
except ModuleNotFoundError:
    has_medmnist = False
import cyy_torch_toolbox.datasets.vision as local_vision_datasets
from cyy_torch_toolbox.ml_type import DatasetType


def get_dataset_constructors(dataset_type: DatasetType = None) -> dict:
    repositories = []
    if dataset_type is None or dataset_type == DatasetType.Vision:
        repositories += [torchvision.datasets, local_vision_datasets]
    if dataset_type is None or dataset_type == DatasetType.Text:
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
        for name in dir(repository):
            dataset_constructor = getattr(repository, name)
            if not inspect.isclass(dataset_constructor):
                continue
            if issubclass(dataset_constructor, torch.utils.data.Dataset):
                dataset_constructors[name] = dataset_constructor
    if has_medmnist and (dataset_type is None or dataset_type == DatasetType.Vision):
        INFO = medmnist.info.INFO
        for name, item in INFO.items():
            medmnist_cls = getattr(medmnist, item["python_class"])
            setattr(medmnist_cls, "targets", item["label"])
            dataset_constructors[name] = functools.partial(
                medmnist_cls, target_transform=lambda x: x[0]
            )

    return dataset_constructors
