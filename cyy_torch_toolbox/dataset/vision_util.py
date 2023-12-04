import functools
import os

import torch
import torch.utils.data
import torchvision

from ..ml_type import DatasetType
from .util import DatasetUtil, global_dataset_util_factor


class VisionDatasetUtil(DatasetUtil):
    @functools.cached_property
    def channel(self):
        x = self._get_sample_input(0)
        assert x.shape[0] <= 3
        return x.shape[0]

    def get_mean_and_std(self):
        if self._name.lower() == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            return (mean, std)
        mean = torch.zeros(self.channel)
        for index in range(len(self)):
            x = self._get_sample_input(index)
            for i in range(self.channel):
                mean[i] += x[i, :, :].mean()
        mean.div_(len(self))

        wh = None
        std = torch.zeros(self.channel)
        for index in range(len(self)):
            x = self._get_sample_input(index)
            if wh is None:
                wh = x.shape[1] * x.shape[2]
            for i in range(self.channel):
                std[i] += torch.sum((x[i, :, :] - mean[i].data.item()) ** 2) / wh
        std = std.div(len(self)).sqrt()
        return mean, std

    def save_sample_image(self, index: int, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sample_input = self._get_sample_input(index, apply_transform=False)
        if "image" in sample_input.__class__.__name__.lower():
            return sample_input
        torchvision.utils.save_image(sample_input, path)


global_dataset_util_factor.register(DatasetType.Vision, VisionDatasetUtil)
