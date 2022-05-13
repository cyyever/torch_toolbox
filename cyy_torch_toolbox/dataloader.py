import copy
import os

import torch
import torchvision
from cyy_naive_lib.log import get_logger

try:
    import nvidia.dali
    from nvidia.dali.pipeline import pipeline_def
    from nvidia.dali.plugin.pytorch import (DALIClassificationIterator,
                                            LastBatchPolicy)

    has_dali = True
except ModuleNotFoundError:
    has_dali = False

from .dataset import get_dataset_size
from .dataset_collection import DatasetCollection
from .hyper_parameter import HyperParameter
from .ml_type import DatasetType, MachineLearningPhase, ModelType

if has_dali:

    @pipeline_def
    def create_dali_pipeline(
        dc: DatasetCollection,
        phase: MachineLearningPhase,
        hyper_parameter: HyperParameter,
        device,
        stream,
    ):
        original_dataset = dc.get_original_dataset(phase)
        is_external_source = False
        if isinstance(original_dataset, torchvision.datasets.folder.ImageFolder):
            dataset = dc.get_dataset(phase)
            samples = original_dataset.samples
            if hasattr(dataset, "indices"):
                samples = [samples[idx] for idx in dataset.indices]
            image_files, labels = nvidia.dali.fn.readers.file(
                files=[s[0] for s in samples],
                labels=[s[1] for s in samples],
                random_shuffle=(phase == MachineLearningPhase.Training),
            )
        else:
            raise RuntimeError(f"unsupported dataset type {type(original_dataset)}")

        raw_transforms = dc.get_transforms(phase=phase)
        raw_transforms = [
            t
            for t in raw_transforms
            if not isinstance(t, torchvision.transforms.transforms.ToTensor)
        ]
        get_logger().debug("raw_transforms are %s", raw_transforms)
        horizontal_mirror = False
        mean_and_std = None
        new_transforms = []
        for idx, transform in enumerate(raw_transforms):
            if isinstance(
                transform, torchvision.transforms.transforms.RandomResizedCrop
            ):
                # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
                device_memory_padding = 211025920 if device is not None else 0
                host_memory_padding = 140544512 if device is not None else 0
                # disable nvJPEG to avoid leak
                images = nvidia.dali.fn.decoders.image_random_crop(
                    image_files,
                    device="cpu" if device is None else "mixed",
                    device_memory_padding=device_memory_padding,
                    host_memory_padding=host_memory_padding,
                    num_attempts=5,
                    hw_decoder_load=0,
                    preallocate_width_hint=0,
                    preallocate_height_hint=0,
                )
                images = nvidia.dali.fn.resize(
                    images,
                    dtype=nvidia.dali.types.FLOAT,
                    device="cpu" if device is None else "gpu",
                    resize_x=transform.size[0],
                    resize_y=transform.size[1],
                )
                continue
            if isinstance(
                transform, torchvision.transforms.transforms.RandomHorizontalFlip
            ):
                horizontal_mirror = nvidia.dali.fn.random.coin_flip(
                    probability=transform.p
                )
                continue
            if isinstance(transform, torchvision.transforms.transforms.Normalize):
                mean_and_std = (copy.deepcopy(transform.mean), transform.std)
                continue
            new_transforms.append(transform)
        assert mean_and_std is not None
        scale = 1.0
        if not is_external_source:
            for idx, m in enumerate(mean_and_std[0]):
                assert 0 <= m <= 1
                mean_and_std[0][idx] = m * 255
            scale = 1.0 / 255
        images = nvidia.dali.fn.crop_mirror_normalize(
            images,
            dtype=nvidia.dali.types.FLOAT,
            output_layout="CHW",
            mean=mean_and_std[0].tolist(),
            std=mean_and_std[1].tolist(),
            scale=scale,
            mirror=horizontal_mirror,
        )

        if new_transforms:
            get_logger().error("remaining raw_transforms are %s", new_transforms)
        assert not new_transforms

        if device is not None:
            labels = labels.gpu()
        return images, labels


def get_dataloader(
    dc: DatasetCollection,
    model_type: ModelType,
    phase: MachineLearningPhase,
    hyper_parameter: HyperParameter,
    device=None,
    stream=None,
    persistent_workers=True,
):
    dataset = dc.get_dataset(phase)
    original_dataset = dc.get_original_dataset(phase)
    if (
        has_dali
        and dc.dataset_type == DatasetType.Vision
        and model_type == ModelType.Classification
        # We use DALI for ImageFolder only
        and isinstance(original_dataset, torchvision.datasets.folder.ImageFolder)
    ):
        get_logger().info("use DALI")
        device_id = -1
        if device is not None:
            device_id = device.index
        pipeline = create_dali_pipeline(
            batch_size=hyper_parameter.batch_size,
            num_threads=2,
            py_start_method="spawn",
            device_id=device_id,
            dc=dc,
            phase=phase,
            hyper_parameter=hyper_parameter,
            device=device,
            stream=stream,
        )
        pipeline.build()
        return DALIClassificationIterator(
            pipeline,
            size=get_dataset_size(dataset),
            auto_reset=True,
            dynamic_shape=True,
            last_batch_policy=LastBatchPolicy.PARTIAL,
            last_batch_padded=True,
        )
    transforms = dc.get_transforms(phase=phase)
    collate_fn = transforms.collate_batch
    if transforms.has_transform() and "USE_PROCESS_DATALOADER" in os.environ:
        num_workers = 2
        prefetch_factor = 1
    else:
        get_logger().info("no using workers")
        num_workers = 0
        persistent_workers = False
        prefetch_factor = 2
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=hyper_parameter.batch_size,
        shuffle=(phase == MachineLearningPhase.Training),
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        collate_fn=collate_fn,
    )
