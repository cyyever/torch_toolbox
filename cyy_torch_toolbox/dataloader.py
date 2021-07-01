import torch
import torchtext

from dataset_collection import DatasetCollection
from hyper_parameter import HyperParameter
from ml_type import DatasetType, MachineLearningPhase


def get_dataloader(
    dc: DatasetCollection,
    phase: MachineLearningPhase,
    hyper_parameter: HyperParameter,
    device=None,
):
    dataset = dc.get_dataset(phase)
    if dc.dataset_type != DatasetType.Text:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=hyper_parameter.batch_size,
            shuffle=(phase == MachineLearningPhase.Training),
            num_workers=2,
            prefetch_factor=1,
            persistent_workers=False,
            pin_memory=True,
        )
    return torchtext.legacy.data.BucketIterator.splits(
        [dataset],
        batch_size=hyper_parameter.batch_size,
        shuffle=(phase == MachineLearningPhase.Training),
        sort_within_batch=True,
        device=device,
    )[0]
