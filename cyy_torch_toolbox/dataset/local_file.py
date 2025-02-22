import os

from datasets import load_dataset


def load_local_files(files: str | list[str]):
    if isinstance(files, str):
        files = [files]
    path = os.path.splitext(files[0])[1][1:]
    return load_dataset(path=path, data_files=files, split="train", cache_dir=None)
