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


from cyy_torch_toolbox.dataset import get_dataset_size
from cyy_torch_toolbox.dataset_collection import DatasetCollection
from cyy_torch_toolbox.hyper_parameter import HyperParameter
from cyy_torch_toolbox.ml_type import MachineLearningPhase, ModelType

if has_dali:

    @pipeline_def
    def create_dali_pipeline(
        dc: DatasetCollection,
        phase: MachineLearningPhase,
        device,
    ):
        original_dataset = dc.get_original_dataset(phase)
        if not isinstance(original_dataset, torchvision.datasets.folder.ImageFolder):
            raise RuntimeError(f"unsupported dataset type {type(original_dataset)}")
        dataset = dc.get_dataset(phase)
        transforms = dc.get_transforms(phase=phase)
        samples = original_dataset.samples
        if hasattr(dataset, "indices"):
            samples = [samples[idx] for idx in dataset.indices]
        files = [s[0] for s in samples]
        labels = [s[1] for s in samples]
        labels = [transforms.transform_target(label) for label in labels]
        image_files, labels = nvidia.dali.fn.readers.file(
            files=files,
            labels=labels,
            random_shuffle=(phase == MachineLearningPhase.Training),
        )

        raw_input_transforms = transforms.get_input_transforms_in_order()
        raw_input_transforms = [
            t
            for t in raw_input_transforms
            if not isinstance(t, torchvision.transforms.ToTensor)
        ]
        get_logger().debug("raw_input_transforms are %s", raw_input_transforms)
        horizontal_mirror = False
        mean_and_std = None
        remain_transforms = []
        images = None
        # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
        device_memory_padding = 211025920 if device is not None else 0
        host_memory_padding = 140544512 if device is not None else 0
        for transform in raw_input_transforms:
            match transform:
                case torchvision.transforms.Resize():
                    if images is None:
                        images = nvidia.dali.fn.decoders.image(
                            image_files,
                            device="cpu" if device is None else "mixed",
                            device_memory_padding=device_memory_padding,
                            host_memory_padding=host_memory_padding,
                        )
                    match transform.size:
                        case int():
                            images = nvidia.dali.fn.resize(
                                images,
                                dtype=nvidia.dali.types.FLOAT,
                                device="cpu" if device is None else "gpu",
                                resize_shorter=transform.size,
                            )
                        case _:
                            images = nvidia.dali.fn.resize(
                                images,
                                dtype=nvidia.dali.types.FLOAT,
                                device="cpu" if device is None else "gpu",
                                resize_x=transform.size[0],
                                resize_y=transform.size[1],
                            )
                case torchvision.transforms.CenterCrop():
                    images = nvidia.dali.fn.crop(
                        images,
                        device="cpu" if device is None else "gpu",
                        crop=transform.size,
                    )
                case torchvision.transforms.RandomResizedCrop():
                    assert images is None
                    images = nvidia.dali.fn.decoders.image_random_crop(
                        image_files,
                        device="cpu" if device is None else "mixed",
                        device_memory_padding=device_memory_padding,
                        host_memory_padding=host_memory_padding,
                    )
                    images = nvidia.dali.fn.resize(
                        images,
                        dtype=nvidia.dali.types.FLOAT,
                        device="cpu" if device is None else "gpu",
                        resize_x=transform.size[0],
                        resize_y=transform.size[1],
                    )
                case torchvision.transforms.RandomHorizontalFlip():
                    horizontal_mirror = nvidia.dali.fn.random.coin_flip(
                        probability=transform.p
                    )
                case torchvision.transforms.Normalize():
                    mean_and_std = (transform.mean, transform.std)
                case _:
                    remain_transforms.append(transform)
        assert mean_and_std is not None
        new_mean = copy.deepcopy(mean_and_std[0])
        for idx, m in enumerate(mean_and_std[0]):
            assert 0 <= m <= 1
            new_mean[idx] = m * 255
        scale = 1.0 / 255
        images = nvidia.dali.fn.crop_mirror_normalize(
            images,
            dtype=nvidia.dali.types.FLOAT,
            output_layout="CHW",
            mean=new_mean.tolist(),
            std=mean_and_std[1].tolist(),
            scale=scale,
            mirror=horizontal_mirror,
        )

        if remain_transforms:
            get_logger().error("remaining input transforms are %s", remain_transforms)
            raise RuntimeError("can't cover all input transforms")

        if device is not None:
            labels = labels.gpu()
        return images, labels


def get_dataloader(
    dc: DatasetCollection,
    model_type: ModelType,
    phase: MachineLearningPhase,
    hyper_parameter: HyperParameter,
    device=None,
    cache_transforms=None,
    use_dali=True,
):
    match cache_transforms:
        case "cpu" | True:
            dc.cache_transforms(phase=phase)
        case "cuda":
            dc.cache_transforms(phase=phase, device=device)

    dataset = dc.get_dataset(phase)
    original_dataset = dc.get_original_dataset(phase)
    # We use DALI for ImageFolder only
    if (
        has_dali
        and use_dali
        and model_type == ModelType.Classification
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
            device=device,
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
    persistent_workers = True
    pin_memory = str(device) != "cpu"
    pin_memory_device = ""
    if pin_memory:
        pin_memory_device = str(device)
    if not dc.transforms_cached(phase=phase) or "USE_PROCESS_DATALOADER" in os.environ:
        num_workers = 2
        prefetch_factor = 1
    else:
        get_logger().debug("use threads")
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
        pin_memory=pin_memory,
        pin_memory_device=pin_memory_device,
        collate_fn=collate_fn,
    )
