import os
import pickle

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy
import PIL
import torch
import torchvision


class DatasetToMelSpectrogram(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        root: str,
        target_index: int = -1,
    ):
        super().__init__(root=root)
        self.__dataset = dataset
        self.__target_index = target_index

    def __getitem__(self, index):
        pickled_file = os.path.join(self.root, "{}.pick".format(index))
        if os.path.exists(pickled_file):
            with open(pickled_file, "rb") as f:
                image_path, target = pickle.load(f)
        else:
            result = self.__dataset.__getitem__(index)
            # we assume sample rate is in slot 1
            tensor = result[0]
            sample_rate = result[1]
            target = result[self.__target_index]
            assert len(tensor.shape) == 2 and tensor.shape[0] == 1
            spectrogram = librosa.feature.melspectrogram(
                tensor[0].numpy(), sr=sample_rate
            )
            log_spectrogram = librosa.power_to_db(spectrogram, ref=numpy.max)
            librosa.display.specshow(
                log_spectrogram, sr=sample_rate, x_axis="time", y_axis="mel"
            )
            image_path = os.path.join(self.root, "{}.png".format(index))
            plt.gcf().savefig(image_path, dpi=50)
            with open(pickled_file, "wb") as f:
                pickle.dump((image_path, target), f)
        img = PIL.Image.open(image_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return (img, target)

    def __len__(self):
        return self.__dataset.__len__()
