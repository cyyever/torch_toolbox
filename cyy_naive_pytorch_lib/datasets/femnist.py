import os
import os.path
import shutil
from typing import Any, Callable, Optional, Tuple

import torch
from PIL import Image
from torchvision import VisionDataset
from torchvision.utils import download_and_extract_archive


class FEMNIST(VisionDataset):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.

    Args:
        root (string): Root directory of dataset where ``FEMNIST/processed/training.pt``
            and  ``FEMNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    resources = [
        (
            "https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz",
            "59c65cec646fc57fe92d27d83afdf0ed",
        )
    ]

    training_file = "training.pt"
    test_file = "test.pt"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets, self.users = torch.load(
            os.path.join(self.processed_folder, data_file)
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], FEMNIST.__relabel_class(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="F")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, {"user": self.users[index]}

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @staticmethod
    def __relabel_class(c):
        """
        maps hexadecimal class value (string) to a decimal number
        returns:
        - 0 through 9 for classes representing respective numbers
        - 10 through 35 for classes representing respective uppercase letters
        - 36 through 61 for classes representing respective lowercase letters
        """
        if c.isdigit() and int(c) < 40:
            return int(c) - 30
        if int(c, 16) <= 90:  # uppercase
            return int(c, 16) - 55
        return int(c, 16) - 61

    def _check_exists(self) -> bool:
        return os.path.exists(
            os.path.join(self.processed_folder, self.training_file)
        ) and os.path.exists(os.path.join(self.processed_folder, self.test_file))

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition("/")[2]
            download_and_extract_archive(
                url, download_root=self.raw_folder, filename=filename, md5=md5
            )

        # process and save as torch files
        print("Processing...")

        shutil.move(
            os.path.join(self.raw_folder, self.training_file), self.processed_folder
        )
        shutil.move(
            os.path.join(self.raw_folder, self.test_file), self.processed_folder
        )
        print("Done!")

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")
