import copy

import numpy
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torch
import torchtext
import torchvision
from cyy_naive_lib.log import get_logger
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

from dataset_collection import DatasetCollection
from hyper_parameter import HyperParameter
from ml_type import DatasetType, MachineLearningPhase, ModelType


class ExternalInputIterator:
    def __init__(self, dataset, batch_size: int, shuffle: bool):
        self.__dataset = copy.deepcopy(dataset)
        if hasattr(self.__dataset, "transform"):
            setattr(self.__dataset, "transform", None)
        if hasattr(self.__dataset, "transforms"):
            setattr(self.__dataset, "transforms", None)
        self.__batch_size = batch_size
        self.__shuffle = shuffle
        self.__batch_sampler = None
        self.__transform = torchvision.transforms.ToTensor()

    def __iter__(self):
        if self.__shuffle:
            sampler = RandomSampler(self.__dataset)
        else:
            sampler = SequentialSampler(self.__dataset)
        self.__batch_sampler = BatchSampler(sampler, self.__batch_size, drop_last=False)
        return self

    def __next__(self):
        batch_iter = self.__batch_sampler.__iter__()
        batch = []
        labels = []
        for idx in batch_iter.__next__():
            sample, label = self.__dataset[idx]
            sample = self.__transform(sample)
            batch.append(sample)
            labels.append(label)
        return (batch, torch.LongTensor(labels))

    def __len__(self):
        return len(self.__dataset)


def get_raw_transformers(obj) -> list:
    raw_transforms = []
    if isinstance(obj, torchvision.transforms.Compose):
        for transform in obj.transforms:
            raw_transforms += get_raw_transformers(transform)
    elif isinstance(obj, torchvision.datasets.vision.StandardTransform):
        raw_transforms += get_raw_transformers(obj.transform)
    else:
        raw_transforms.append(obj)
    assert raw_transforms
    return raw_transforms


@pipeline_def
def create_dali_pipeline(
    dc: DatasetCollection,
    phase: MachineLearningPhase,
    hyper_parameter: HyperParameter,
    device,
):
    dataset = dc.get_dataset(phase)
    original_dataset = dc.get_original_dataset(phase)
    if isinstance(original_dataset, torchvision.datasets.folder.ImageFolder):
        samples = original_dataset.samples
        if hasattr(dataset, "indices"):
            get_logger().info("use indices")
            samples = [samples[idx] for idx in dataset.indices]
        image_files, labels = fn.readers.file(
            files=[s[0] for s in samples],
            labels=[s[1] for s in samples],
            random_shuffle=(phase == MachineLearningPhase.Training),
        )
    else:
        external_source = ExternalInputIterator(
            dataset,
            batch_size=hyper_parameter.batch_size,
            shuffle=(phase == MachineLearningPhase.Training),
        )
        images, labels = fn.external_source(
            source=external_source, layout=["CHW", ""], num_outputs=2
        )

    raw_transforms = get_raw_transformers(dataset.transforms)
    raw_transforms = [
        t
        for t in raw_transforms
        if not isinstance(t, torchvision.transforms.transforms.ToTensor)
    ]
    raw_transform_dict = {
        idx: transform for idx, transform in enumerate(raw_transforms)
    }
    get_logger().info("raw_transforms are %s", raw_transform_dict)
    crop_size = None
    horizontal_mirror = False
    mean_and_std = None
    raw_transform_dict_copy = copy.copy(raw_transform_dict)
    for idx in sorted(raw_transform_dict_copy):
        transform = raw_transform_dict_copy[idx]
        if isinstance(transform, torchvision.transforms.transforms.RandomResizedCrop):
            images = fn.decoders.image_random_crop(
                image_files,
                device="cpu" if device is None else "mixed",
                num_attempts=5,
            )
            images = fn.resize(
                images,
                dtype=types.FLOAT,
                device="cpu" if device is None else "gpu",
                resize_x=transform.size[0],
                resize_y=transform.size[1],
            )
            crop_size = transform.size
            raw_transform_dict.pop(idx)
            continue
        if isinstance(
            transform, torchvision.transforms.transforms.RandomHorizontalFlip
        ):
            horizontal_mirror = fn.random.coin_flip(probability=transform.p)
            raw_transform_dict.pop(idx)
            continue
        if isinstance(transform, torchvision.transforms.transforms.Normalize):
            mean_and_std = (transform.mean, transform.std)
            raw_transform_dict.pop(idx)
            continue
    assert mean_and_std is not None
    if crop_size is None:
        crop_size = (10000000000, 1000000000000)
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        crop_h=crop_size[0],
        crop_w=crop_size[1],
        mean=mean_and_std[0].tolist(),
        std=mean_and_std[1].tolist(),
        mirror=horizontal_mirror,
    )

    if raw_transform_dict:
        get_logger().info("remaining raw_transforms are %s", raw_transform_dict)
    assert not raw_transform_dict

    if device is not None:
        labels = labels.gpu()
    return images, labels


def get_dataloader(
    dc: DatasetCollection,
    model_type: ModelType,
    phase: MachineLearningPhase,
    hyper_parameter: HyperParameter,
    device=None,
):
    dataset = dc.get_dataset(phase)
    if dc.dataset_type == DatasetType.Vision and model_type == ModelType.Classification:
        get_logger().info("use DALI")
        device_id = -1
        if device is not None:
            device_id = device.index
        pipeline = create_dali_pipeline(
            batch_size=hyper_parameter.batch_size,
            # num_threads=os.cpu_count(),
            num_threads=2,
            device_id=device_id,
            dc=dc,
            phase=phase,
            hyper_parameter=hyper_parameter,
            device=device,
        )
        pipeline.build()
        return DALIClassificationIterator(pipeline, auto_reset=True)
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
