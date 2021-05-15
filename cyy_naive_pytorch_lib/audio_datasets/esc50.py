import os
import os.path
from pathlib import Path

import torchaudio
from torch.utils.data import Dataset
from torchvision.datasets.utils import (download_and_extract_archive,
                                        verify_str_arg)


class ESC50(Dataset):
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    """
    The ESC-50 dataset is a labeled collection of 2000 environmental audio recordings suitable for benchmarking methods of environmental sound classification.

    The dataset consists of 5-second-long recordings organized into 50 semantical classes (with 40 examples per class) loosely arranged into 5 major categories:

    Args:
        root (string): Root directory of dataset.
        split (string): The dataset split to use. One of {``train``, ``val``, ``test``}.
            Defaults to ``train``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, root: str, split: str, download: bool):
        self.root = root
        # check arguments
        self.split = verify_str_arg(split, "split", ("train", "val", "test"))

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )

        # Get directory listing from path
        files = Path(self.raw_folder).glob("**/*.wav")
        # Iterate through the listing and create a list of tuples (filename, label)
        self.items = [
            (str(f), f.name.split("-")[-1].replace(".wav", "")) for f in files
        ]
        if self.split == "train":
            self.items = [
                p
                for p in self.items
                if (
                    os.path.basename(p[0]).startswith("1")
                    or os.path.basename(p[0]).startswith("2")
                    or os.path.basename(p[0]).startswith("3")
                )
            ]
        elif self.split == "val":
            self.items = [
                p for p in self.items if os.path.basename(p[0]).startswith("4")
            ]
        else:
            self.items = [
                p for p in self.items if os.path.basename(p[0]).startswith("5")
            ]

        self.length = len(self.items)

    def __getitem__(self, index):
        filename, label = self.items[index]
        tensor, _ = torchaudio.load(filename)
        return (tensor, int(label))

    def __len__(self):
        return self.length

    def download(self) -> None:
        """Download data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        filename = self.url.rpartition("/")[2]
        download_and_extract_archive(
            self.url, download_root=self.raw_folder, filename=filename
        )

    def _check_exists(self) -> bool:
        return os.path.exists(self.raw_folder)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")
